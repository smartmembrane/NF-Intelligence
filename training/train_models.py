"""
train_models.py —— 模型训练与导出（v3）

特性：
1. 数据来源：自动合并 data/ 目录下所有 .xlsx（列结构需一致）→ 直接把新数据表复制进 data/ 即可扩充数据集
2. 每个目标在 ExtraTrees / HistGBR / 两者集成 中按 KFold 交叉验证自动选最优（提升拟合度）
3. 分位数回归输出 10%/90% 预测区间
4. 训练结果输出到 results/<时间戳>_train/ 文件夹（metrics.xlsx + 完整 json 记录，
   其中保留 GroupKFold 外推指标存档备查，界面不显示）
5. 模型包内记录数据指纹，数据变化时服务启动会自动重训
"""
import argparse, json, os, sys, glob, hashlib, datetime
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import joblib
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              GradientBoostingRegressor, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (KFold, GroupKFold, GroupShuffleSplit,
                                     cross_val_predict)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from app.nf_shap import tree_shap, shap_summary
from app.nf_features import (recipe_only_features, STAGE1_CHARS, stage1_fwd, stage1_inv,
    
    TARGETS, MEMBRANE_COL, SMILES_COLS, DENS_COLS, TIME_COLS, CHAR_COLS, OP_COLS,
    fwd_transform, inv_transform, build_feature_matrix, build_synth_key, EnsembleModel)

RANDOM_STATE = 42

# ---- 模型适用域（物理阈值过滤，透明记录）----
# 仅排除极端杠杆离群点；Mg/Li 截留保持全量（收窄范围反而降低 R²，且失败案例有信息价值）
DOMAIN_FILTERS = {
    "Flux_Mg": (None, 40.0),   # 排除 >40 LMH 的极端点（数据中仅1个 96 LMH）
    "Flux_Li": (None, 30.0),   # 排除 >30 LMH 的极端点（3个）
}
DATA_DIR = os.path.join(_ROOT, "data")


# ---------------- 数据加载：合并 data/ 下所有 xlsx ----------------
def data_fingerprint(paths):
    h = hashlib.md5()
    for p in sorted(paths):
        st = os.stat(p)
        h.update(f"{os.path.basename(p)}|{st.st_size}|{int(st.st_mtime)}".encode())
    from app.nf_features import RDKIT_AVAILABLE
    h.update(f"|rdkit={RDKIT_AVAILABLE}".encode())
    return h.hexdigest()


def load_all_data():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.xlsx")))
    paths = [p for p in paths if not os.path.basename(p).startswith("~$")]  # 跳过Excel临时文件
    if not paths:
        raise FileNotFoundError(
            f"data/ 目录下没有找到任何 .xlsx 数据文件。请把数据表放入 {DATA_DIR}")
    frames = []
    for p in paths:
        try:
            xl = pd.ExcelFile(p)
            sheet = "main" if "main" in xl.sheet_names else xl.sheet_names[0]
            df = pd.read_excel(p, sheet_name=sheet)
            frames.append(df)
            print(f"  读入 {os.path.basename(p)} [{sheet}] : {len(df)} 行")
        except Exception as e:
            print(f"  跳过 {os.path.basename(p)}（读取失败: {e}）")
    df = pd.concat(frames, ignore_index=True, sort=False)
    # 按合成键+操作条件去重，避免重复复制同一数据表导致样本翻倍
    df["synth_key"] = build_synth_key(df)
    before = len(df)
    df = df.drop_duplicates(subset=["synth_key"] + OP_COLS + list(TARGETS.values()),
                            keep="first").reset_index(drop=True)
    if len(df) < before:
        print(f"  去重: {before} -> {len(df)} 行")
    print(f"合并后总数据: {len(df)} 行, 唯一配方 {df['synth_key'].nunique()} 个")
    return df, data_fingerprint(paths)


# ---------------- 模型 ----------------
def make_et(max_features=0.7):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", ExtraTreesRegressor(n_estimators=400, max_features=max_features,
                                               min_samples_leaf=1, bootstrap=False,
                                               random_state=RANDOM_STATE, n_jobs=1))])

def make_rf():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", RandomForestRegressor(n_estimators=400, max_features=0.5,
                                                 min_samples_leaf=1,
                                                 random_state=RANDOM_STATE, n_jobs=1))])

def make_hgb():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                         max_leaf_nodes=41, min_samples_leaf=10,
                                         l2_regularization=0.5, random_state=RANDOM_STATE))])

def make_quantile(q):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", GradientBoostingRegressor(loss="quantile", alpha=q,
                                                     n_estimators=200, learning_rate=0.06,
                                                     max_depth=3, random_state=RANDOM_STATE))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "models"))
    args, _ = ap.parse_known_args()
    os.makedirs(args.out, exist_ok=True)

    df, fingerprint = load_all_data()
    for c in DENS_COLS + TIME_COLS + CHAR_COLS + OP_COLS + list(TARGETS.values()):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    membrane_categories = sorted(df[MEMBRANE_COL].fillna("Unk").astype(str).unique().tolist())
    FEAT = build_feature_matrix(df.copy(), membrane_categories=membrane_categories)
    feature_names = FEAT.columns.tolist()

    # 时间命名的训练结果文件夹
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = os.path.join(_ROOT, "results", f"{ts}_train")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    report, full_record, models_bundle, importances = {}, {}, {}, {}
    PLOT_DATA = {}
    POINT_DATA = {}
    SHAP_DATA = {}
    shap_export = {}

    excluded_log = {}
    for key, ycol in TARGETS.items():
        mask = df[ycol].notna()
        X = FEAT[mask].values
        y_raw = df.loc[mask, ycol].values
        # 适用域过滤（阈值与排除数记录在训练存档）
        orig_idx = df.index[mask].to_numpy()
        if key in DOMAIN_FILTERS:
            lo, hi = DOMAIN_FILTERS[key]
            keep = np.ones(len(y_raw), bool)
            if lo is not None: keep &= y_raw >= lo
            if hi is not None: keep &= y_raw <= hi
            excluded_log[key] = {"rule": f"[{lo}, {hi}]", "n_excluded": int(np.sum(~keep))}
            X, y_raw = X[keep], y_raw[keep]
            g_keep = df.loc[mask, "synth_key"].values[keep]
            orig_idx = orig_idx[keep]
        else:
            g_keep = None
        y = fwd_transform(key, y_raw)
        groups = g_keep if g_keep is not None else df.loc[mask, "synth_key"].values
        n_groups = len(np.unique(groups))

        # ---- 候选模型按 KFold OOF 自动选优 + 最优两模型加权混合 ----
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        makers = {"ET.5": lambda: make_et(0.5), "ET.7": lambda: make_et(0.7),
                  "ET1":  lambda: make_et(1.0), "HGB": make_hgb, "RF": make_rf}
        oof = {nm: cross_val_predict(mk(), X, y, cv=kf, n_jobs=1)
               for nm, mk in makers.items()}
        cand = {nm: r2_score(y, p) for nm, p in oof.items()}
        top2 = sorted(cand, key=cand.get, reverse=True)[:2]
        best_w, best_r = 1.0, cand[top2[0]]
        for w in np.arange(0.5, 1.01, 0.05):        # 混合权重网格
            r = r2_score(y, w*oof[top2[0]] + (1-w)*oof[top2[1]])
            if r > best_r: best_r, best_w = r, float(w)
        best_name = top2[0] if best_w >= 0.999 else f"{top2[0]}+{top2[1]}"
        kf_r2 = best_r
        blend = (top2, best_w)

        # ---- GroupKFold 外推指标（仅存档，不在界面显示；只算选中的模型以省时）----
        gkf = GroupKFold(n_splits=min(5, n_groups))
        (t2, w) = blend
        pg1 = cross_val_predict(makers[t2[0]](), X, y, cv=gkf, groups=groups, n_jobs=1)
        if w >= 0.999:
            pg = pg1
        else:
            pg2 = cross_val_predict(makers[t2[1]](), X, y, cv=gkf, groups=groups, n_jobs=1)
            pg = w*pg1 + (1-w)*pg2
        gkf_r2 = r2_score(y, pg)

        # ---- 随机留出测试集：5 次不同划分取平均（消除小测试集单次抽样波动）----
        from sklearn.model_selection import train_test_split
        def fit_predict(Xtr, ytr, Xte):
            (t2, w) = blend
            p = w * makers[t2[0]]().fit(Xtr, ytr).predict(Xte)
            if w < 0.999:
                p = p + (1-w) * makers[t2[1]]().fit(Xtr, ytr).predict(Xte)
            return p
        r2s, rmses = [], []
        pool_true, pool_pred, pool_idx, pool_seed = [], [], [], []
        idx = np.arange(len(y))
        for seed in range(5):
            tr, te = train_test_split(idx, test_size=0.2, random_state=seed, shuffle=True)
            pred = inv_transform(key, fit_predict(X[tr], y[tr], X[te]))
            r2s.append(r2_score(y_raw[te], pred))
            rmses.append(float(np.sqrt(mean_squared_error(y_raw[te], pred))))
            pool_true.append(y_raw[te]); pool_pred.append(pred)
            pool_idx.append(orig_idx[te]); pool_seed.append(np.full(len(te), seed))
        test_r2 = float(np.mean(r2s))
        test_r2_std = float(np.std(r2s))
        test_rmse = float(np.mean(rmses))
        pool_true = np.concatenate(pool_true); pool_pred = np.concatenate(pool_pred)
        pool_idx = np.concatenate(pool_idx); pool_seed = np.concatenate(pool_seed)
        test_mae = float(mean_absolute_error(pool_true, pool_pred))
        PLOT_DATA[key] = (pool_true, pool_pred)
        POINT_DATA[key] = (pool_idx, pool_seed, pool_true, pool_pred)

        # ---- 全数据重训最终模型 ----
        (t2, w) = blend
        m1 = makers[t2[0]]().fit(X, y)
        if w >= 0.999:
            final = EnsembleModel([m1]); final.weights = [1.0]
        else:
            m2 = makers[t2[1]]().fit(X, y)
            final = EnsembleModel([m1, m2]); final.weights = [w, 1-w]
        et_full = m1  # 供重要性计算

        q_lo = make_quantile(0.1).fit(X, y)
        q_hi = make_quantile(0.9).fit(X, y)

        try:
            pi = permutation_importance(et_full, X, y, n_repeats=4,
                                        random_state=RANDOM_STATE, n_jobs=1)
            order = np.argsort(pi.importances_mean)[::-1][:15]
            importances[key] = [{"feature": feature_names[i],
                                 "importance": float(pi.importances_mean[i])} for i in order]
        except Exception:
            importances[key] = []

        # ---- SHAP 分析（TreeSHAP，基于选中模型的第一棵树集成）----
        try:
            sh = tree_shap(m1, X, feature_names=feature_names,
                           max_samples=min(200, len(X)), random_state=RANDOM_STATE)
            fn_kept = [feature_names[i] for i in sh["kept"]]
            sh["feature_names"] = fn_kept
            SHAP_DATA[key] = sh
            shap_export[key] = shap_summary(sh["shap_values"], fn_kept,
                                            sh["X_used"], top_k=20)
            print(f"         SHAP: base={sh['base_value']:.3f}, "
                  f"top1={shap_export[key][0]['feature']}")
        except Exception as e:
            print(f"         SHAP 计算跳过: {e}")
            shap_export[key] = []

        models_bundle[key] = {"model": final, "q_lo": q_lo, "q_hi": q_hi,
                              "model_type": best_name}
        # 界面展示用（不含 GroupKFold）
        report[key] = {"n": int(len(y_raw)), "n_groups": int(n_groups),
                       "CV_R2": round(float(kf_r2), 3),
                       "Test_R2": round(float(test_r2), 3),
                       "RMSE": round(test_rmse, 3),
                       "MAE": round(test_mae, 3),
                       "model": best_name}
        # 完整存档（含 GroupKFold 外推指标与三模型对比）
        full_record[key] = dict(report[key], Test_R2_std=round(test_r2_std, 3),
                                GroupKFold_R2=round(float(gkf_r2), 3),
                                candidates={k: round(v, 3) for k, v in cand.items()})
        print(f"{key:8s} n={mask.sum():4d} 选用={best_name:3s} "
              f"CV_R2={kf_r2:.3f} Test_R2={test_r2:.3f} RMSE={test_rmse:.3f}")

    # ---- 第一阶段：配方 → 表征（两阶段建模）----
    print("\n[两阶段] 训练第一阶段：配方 → 膜表征")
    Xrec = recipe_only_features(df.copy(), membrane_categories).values
    rec_feature_names = recipe_only_features(df.copy(), membrane_categories).columns.tolist()
    stage1_bundle = {}
    stage1_report = {}
    kf1 = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for col, tf in STAGE1_CHARS.items():
        yv = pd.to_numeric(df[col], errors="coerce").values
        mrec = ~np.isnan(yv)
        if mrec.sum() < 40:
            continue
        yt = stage1_fwd(col, yv[mrec])
        def _mk():
            return Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("m", ExtraTreesRegressor(n_estimators=400, max_features=0.6,
                                    min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=1))])
        oof = cross_val_predict(_mk(), Xrec[mrec], yt, cv=kf1, n_jobs=1)
        r2c = r2_score(yv[mrec], stage1_inv(col, oof))
        final1 = _mk().fit(Xrec[mrec], yt)
        stage1_bundle[col] = final1
        stage1_report[col] = {"n": int(mrec.sum()), "R2": round(float(r2c), 3),
                              "transform": tf or "none"}
        print(f"    {col:24s} n={int(mrec.sum()):3d}  R2={r2c:.3f}")

    # 保存第一阶段结果到 results
    with open(os.path.join(RESULTS_DIR, "stage1_recipe_to_char.json"), "w", encoding="utf-8") as f:
        json.dump(stage1_report, f, ensure_ascii=False, indent=2)

    # ---- 模型落盘 ----
    joblib.dump({"models": models_bundle, "feature_names": feature_names,
                 "membrane_categories": membrane_categories,
                 "targets": list(TARGETS.keys()),
                 "stage1_models": stage1_bundle,
                 "stage1_feature_names": rec_feature_names,
                 "stage1_report": stage1_report,
                 "data_fingerprint": fingerprint,
                 "trained_at": ts},
                os.path.join(args.out, "nf_models.joblib"))
    with open(os.path.join(args.out, "stage1.json"), "w", encoding="utf-8") as f:
        json.dump(stage1_report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "shap.json"), "w", encoding="utf-8") as f:
        json.dump(shap_export, f, ensure_ascii=False, indent=2)
    for name, obj in [("metrics.json", report), ("importances.json", importances)]:
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    options = {"membranes": membrane_categories,
               "smiles_aq": sorted(df["SMILES_part1"].dropna().astype(str).unique().tolist()),
               "smiles_org": sorted(df["SMILES_part2"].dropna().astype(str).unique().tolist())}
    with open(os.path.join(args.out, "options.json"), "w", encoding="utf-8") as f:
        json.dump(options, f, ensure_ascii=False, indent=2)

    # ---- 训练结果输出到时间命名文件夹 ----
    rec_df = pd.DataFrame(full_record).T
    rec_df.index.name = "target"
    rec_df.to_excel(os.path.join(RESULTS_DIR, "metrics.xlsx"))
    with open(os.path.join(RESULTS_DIR, "train_record.json"), "w", encoding="utf-8") as f:
        json.dump({"trained_at": ts, "n_rows": int(len(df)),
                   "n_recipes": int(df['synth_key'].nunique()),
                   "data_fingerprint": fingerprint,
                   "domain_filters": excluded_log,
                   "targets": full_record,
                   "note": "GroupKFold_R2 为按配方分组的外推指标，存档备查；界面展示 CV_R2(KFold) 与 Test_R2"},
                  f, ensure_ascii=False, indent=2)
    imp_rows = [dict(target=k, **d) for k, v in importances.items() for d in v]
    pd.DataFrame(imp_rows).to_excel(os.path.join(RESULTS_DIR, "feature_importance.xlsx"), index=False)

    # ---- 逐点测试数据导出（每目标一个 sheet，含样本溯源列，便于定位偏离点）----
    id_cols = [c for c in ("ID", "Number", "sample_id", MEMBRANE_COL) if c in df.columns]
    with pd.ExcelWriter(os.path.join(RESULTS_DIR, "test_predictions.xlsx")) as xw:
        for k, (pidx, pseed, pt, pp) in POINT_DATA.items():
            out = df.loc[pidx, id_cols].copy().reset_index(drop=True)
            out.insert(0, "split_seed", pseed)
            out["y_true"] = pt
            out["y_pred"] = np.round(pp, 3)
            out["residual"] = np.round(pp - pt, 3)
            out["abs_error"] = np.round(np.abs(pp - pt), 3)
            out = out.sort_values("abs_error", ascending=False).reset_index(drop=True)
            out.to_excel(xw, sheet_name=k[:31], index=False)
    print("逐点测试数据已导出: test_predictions.xlsx（按绝对误差降序，便于定位偏离样本）")

    # ---- SHAP 数据导出 ----
    with pd.ExcelWriter(os.path.join(RESULTS_DIR, "shap_analysis.xlsx")) as xw:
        for k, rows in shap_export.items():
            if rows:
                pd.DataFrame(rows).to_excel(xw, sheet_name=f"{k[:24]}_summary", index=False)
        for k, sh in SHAP_DATA.items():
            sv = pd.DataFrame(sh["shap_values"], columns=sh["feature_names"])
            keep = sv.abs().mean().sort_values(ascending=False).head(20).index
            sv[keep].to_excel(xw, sheet_name=f"{k[:22]}_values", index=False)
    print("SHAP 数据已导出: shap_analysis.xlsx（各目标汇总表 + 逐样本 SHAP 值）")

    # ---- 可视化输出（matplotlib 可选，缺失时跳过不报错）----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager as fm
        for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                   "/System/Library/Fonts/PingFang.ttc"]:
            if os.path.exists(fp):
                try:
                    fm.fontManager.addfont(fp)
                    plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=fp).get_name(), "DejaVu Sans"]
                    break
                except Exception:
                    pass
        plt.rcParams["axes.unicode_minus"] = False
        UNITS = {"PWP": "LMH/bar", "Mg_rej": "%", "Flux_Mg": "LMH", "Li_rej": "%", "Flux_Li": "LMH"}

        # 图1: 各目标 预测 vs 实测（5次划分合并的测试点）
        keys = [k for k in TARGETS if k in PLOT_DATA]
        fig, axes = plt.subplots(1, len(keys), figsize=(4.2*len(keys), 3.9))
        for ax, k in zip(np.atleast_1d(axes), keys):
            yt, yp = PLOT_DATA[k]
            ax.scatter(yt, yp, s=14, alpha=0.5, color="#0E7C86", edgecolor="none")
            lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            ax.set_title(f"{k}\nR2={report[k]['Test_R2']:.3f}  MAE={report[k]['MAE']:.2f} {UNITS[k]}", fontsize=10)
            ax.set_xlabel("实测"); ax.set_ylabel("预测")
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "fig1_pred_vs_true.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 图2: 残差 vs 实测（诊断个别偏离样本）
        fig, axes = plt.subplots(1, len(keys), figsize=(4.2*len(keys), 3.6))
        for ax, k in zip(np.atleast_1d(axes), keys):
            yt, yp = PLOT_DATA[k]
            ax.scatter(yt, yp - yt, s=14, alpha=0.5, color="#C97F2B", edgecolor="none")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(k, fontsize=10)
            ax.set_xlabel("实测"); ax.set_ylabel("残差(预测-实测)")
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "fig2_residuals.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 图3: 目标分布直方图（展示偏态与离群）
        fig, axes = plt.subplots(1, len(keys), figsize=(4.2*len(keys), 3.3))
        for ax, k in zip(np.atleast_1d(axes), keys):
            yv = pd.to_numeric(df[TARGETS[k]], errors="coerce").dropna()
            ax.hist(yv, bins=30, color="#0E7C86", edgecolor="white")
            ax.set_title(f"{k} 分布 (n={len(yv)})", fontsize=10)
            ax.set_xlabel(UNITS[k]); ax.set_ylabel("样本数")
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "fig3_target_distributions.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 图4: 指标汇总条形图
        fig, ax = plt.subplots(figsize=(8, 4.2))
        xs = np.arange(len(keys))
        ax.bar(xs-0.2, [report[k]["CV_R2"] for k in keys], width=0.4, label="CV R2", color="#0E7C86")
        ax.bar(xs+0.2, [report[k]["Test_R2"] for k in keys], width=0.4, label="Test R2(5次均值)", color="#C97F2B")
        ax.set_xticks(xs); ax.set_xticklabels(keys)
        ax.set_ylim(0, 1); ax.legend(); ax.set_title("模型拟合度汇总")
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "fig4_metrics_summary.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        # 图5: SHAP 重要性条形图（各目标）
        if shap_export:
            ks = [k for k in TARGETS if shap_export.get(k)]
            fig, axes = plt.subplots(1, len(ks), figsize=(4.6*len(ks), 4.4))
            for ax, k in zip(np.atleast_1d(axes), ks):
                rows = shap_export[k][:12][::-1]
                names = [r["feature"] for r in rows]
                vals = [r["mean_abs_shap"] for r in rows]
                cols = ["#0E7C86" if (r["direction_corr"] or 0) >= 0 else "#C97F2B" for r in rows]
                ax.barh(range(len(rows)), vals, color=cols)
                ax.set_yticks(range(len(rows)))
                ax.set_yticklabels(names, fontsize=7)
                ax.set_xlabel("平均 |SHAP|", fontsize=9)
                ax.set_title(f"{k}  SHAP 重要性", fontsize=10)
            plt.tight_layout()
            fig.savefig(os.path.join(RESULTS_DIR, "fig5_shap_importance.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

            # 图6: SHAP 蜂群图（值分布 vs 特征值高低）
            fig, axes = plt.subplots(1, len(ks), figsize=(4.6*len(ks), 4.4))
            for ax, k in zip(np.atleast_1d(axes), ks):
                sh = SHAP_DATA[k]
                sv, Xu = sh["shap_values"], sh["X_used"]
                top = np.argsort(np.abs(sv).mean(axis=0))[::-1][:10][::-1]
                for row, j in enumerate(top):
                    xv = Xu[:, j].astype(float)
                    rng_ = np.ptp(xv)
                    cvals = (xv - xv.min()) / rng_ if rng_ > 1e-12 else np.zeros_like(xv)
                    jitter = (np.random.rand(len(sv)) - 0.5) * 0.32
                    sc = ax.scatter(sv[:, j], row + jitter, c=cvals, cmap="coolwarm",
                                    s=8, alpha=0.65, edgecolor="none")
                ax.axvline(0, color="k", lw=0.7)
                ax.set_yticks(range(len(top)))
                ax.set_yticklabels([sh["feature_names"][j] for j in top], fontsize=7)
                ax.set_xlabel("SHAP 值（对预测的贡献）", fontsize=9)
                ax.set_title(f"{k}  SHAP 蜂群图", fontsize=10)
            cb = fig.colorbar(sc, ax=np.atleast_1d(axes).tolist(), shrink=0.7, pad=0.01)
            cb.set_label("特征值（低→高）", fontsize=8)
            fig.savefig(os.path.join(RESULTS_DIR, "fig6_shap_beeswarm.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"可视化图表已输出 6 张至 {RESULTS_DIR}（含 2 张 SHAP 图）")
        else:
            print(f"可视化图表已输出 4 张至 {RESULTS_DIR}")
    except ImportError:
        print("未安装 matplotlib，跳过可视化输出（pip install matplotlib 后可启用）")

    print(f"\n模型已导出至 {args.out}")
    print(f"训练结果已保存至 {RESULTS_DIR}")


if __name__ == "__main__":
    main()
