"""
FastAPI 后端 —— 纳滤膜 Mg²⁺/Li⁺ 分离性能预测服务

接口：
  GET  /                  前端页面
  GET  /api/options       基底膜/单体下拉选项
  GET  /api/metrics       各目标模型性能（GroupKFold/Test R²）
  GET  /api/importance    特征重要性（按目标）
  POST /api/predict       单配方预测（含预测区间 + 综合得分）
  POST /api/screen        批量扫描 D1/D2/T1，返回 Top-N 配方

运行：
  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import os, json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

import app.nf_features as nf
from app.nf_features import (
    build_feature_matrix, inv_transform, composite_score, TARGETS,
    SMILES_COLS, DENS_COLS, TIME_COLS, CHAR_COLS, OP_COLS, MEMBRANE_COL,
    recipe_only_features, STAGE1_CHARS, stage1_inv)

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "nf_models.joblib")


def _ensure_models():
    """确保模型存在且与当前 scikit-learn 版本兼容。
    若缺失或反序列化失败（常见于不同 sklearn 版本），用本机环境自动重训。
    这样无论用户装的是哪个版本，模型都与之匹配，避免 pickle 版本不兼容报错。
    """
    need_train = not os.path.exists(MODEL_FILE)
    if not need_train:
        try:
            b = joblib.load(MODEL_FILE)  # 试加载，验证版本兼容
            # 数据变化检测：data/ 目录内容与训练时不一致则自动重训
            import glob as _glob, hashlib as _hl
            paths = sorted(p for p in _glob.glob(os.path.join(ROOT, "data", "*.xlsx"))
                           if not os.path.basename(p).startswith("~$"))
            h = _hl.md5()
            for pth in paths:
                st = os.stat(pth)
                h.update(f"{os.path.basename(pth)}|{st.st_size}|{int(st.st_mtime)}".encode())
            from app.nf_features import RDKIT_AVAILABLE as _rk
            h.update(f"|rdkit={_rk}".encode())
            if b.get("data_fingerprint") != h.hexdigest():
                print("[启动] 检测到 data/ 目录数据有变化，将自动重新训练模型。")
                need_train = True
        except Exception as e:
            print(f"[启动] 已有模型无法加载（可能 sklearn 版本不同）：{e}\n[启动] 将用当前环境重新训练。")
            need_train = True
    if need_train:
        print("[启动] 未找到可用模型，正在用打包数据集训练（约 1–2 分钟，仅首次）……")
        import importlib.util
        train_py = os.path.join(ROOT, "training", "train_models.py")
        spec = importlib.util.spec_from_file_location("nf_train", train_py)
        mod = importlib.util.module_from_spec(spec)
        import sys as _sys
        _sys.argv = ["train_models.py"]  # 用默认参数（自动发现数据集）
        spec.loader.exec_module(mod)
        mod.main()
        print("[启动] 训练完成。")


_ensure_models()

# ---- 加载模型与元数据 ----
_bundle = joblib.load(MODEL_FILE)
FEATURE_NAMES = _bundle["feature_names"]
MEMBRANE_CATS = _bundle["membrane_categories"]
MODELS = _bundle["models"]
# 两阶段建模：第一阶段（配方 → 表征）模型，旧模型可能没有
STAGE1_MODELS = _bundle.get("stage1_models", {})
STAGE1_FEATURES = _bundle.get("stage1_feature_names")
STAGE1_REPORT = _bundle.get("stage1_report", {})

with open(os.path.join(MODELS_DIR, "metrics.json"), encoding="utf-8") as f:
    METRICS = json.load(f)
with open(os.path.join(MODELS_DIR, "importances.json"), encoding="utf-8") as f:
    IMPORTANCES = json.load(f)
with open(os.path.join(MODELS_DIR, "options.json"), encoding="utf-8") as f:
    OPTIONS = json.load(f)
try:
    with open(os.path.join(MODELS_DIR, "shap.json"), encoding="utf-8") as f:
        SHAP_SUMMARY = json.load(f)
except Exception:
    SHAP_SUMMARY = {}

# ---- 单体清洗与官能团分流（供潜力单体筛选）----
import re as _re
def _clean_smiles(v):
    if not isinstance(v, str): return False
    v = v.strip()
    if len(v) < 2 or len(v) > 120: return False
    if "{" in v or "}" in v: return False
    if v.count(".") > 2: return False
    return bool(_re.match(r"^[A-Za-z0-9@+\-\[\]\(\)=#/\\.%]+$", v))

def _is_acyl(v):  return "C(=O)Cl" in v or "C(Cl)=O" in v
def _is_amine(v): return ("N" in v or "n" in v) and not _is_acyl(v)

# 扩展单体库：界面聚合常用单体（文献常见，不限于本数据集）
# name 用于界面显示；SMILES 用于特征计算
EXTERNAL_MONOMERS = [
    # ---- 水相多胺 ----
    ("PIP",        "C1CNCCN1",                                  "哌嗪（经典 NF 水相单体）"),
    ("2-MePIP",    "CC1CNCCN1",                                 "2-甲基哌嗪"),
    ("2,5-DMPIP",  "CC1CNC(C)CN1",                              "2,5-二甲基哌嗪"),
    ("AMPIP",      "NCC1CNCCN1",                                "氨甲基哌嗪"),
    ("HPIP",       "OCCN1CCNCC1",                               "羟乙基哌嗪"),
    ("MPD",        "Nc1cccc(N)c1",                              "间苯二胺（RO 经典）"),
    ("PPD",        "Nc1ccc(N)cc1",                              "对苯二胺"),
    ("OPD",        "Nc1ccccc1N",                                "邻苯二胺"),
    ("TAB",        "Nc1cc(N)cc(N)c1",                           "1,3,5-三氨基苯"),
    ("EDA",        "NCCN",                                      "乙二胺"),
    ("DETA",       "NCCNCCN",                                   "二乙烯三胺"),
    ("TETA",       "NCCNCCNCCN",                                "三乙烯四胺"),
    ("TEPA",       "NCCNCCNCCNCCN",                             "四乙烯五胺"),
    ("PDA-1,3",    "NCCCN",                                     "1,3-丙二胺"),
    ("BDA",        "NCCCCN",                                    "1,4-丁二胺"),
    ("HDA",        "NCCCCCCN",                                  "1,6-己二胺"),
    ("JEFFAMINE",  "NCCCOCCOCCCN",                              "醚二胺（柔性链）"),
    ("TRIS",       "NC(CO)(CO)CO",                              "三羟甲基氨基甲烷"),
    ("BAC",        "NCC1CCC(CN)CC1",                            "1,4-环己烷二甲胺"),
    ("IPD",        "CC1(C)CC(N)CC(C)(CN)C1",                    "异佛尔酮二胺"),
    ("DAPP",       "NCCCN1CCN(CCCN)CC1",                        "双氨丙基哌嗪"),
    ("TAEA",       "NCCN(CCN)CCN",                              "三(2-氨乙基)胺"),
    ("PEHA",       "NCCNCCNCCNCCNCCN",                          "五乙烯六胺"),
    ("SPD",        "NCCCNCCCCNCCCN",                            "亚精胺类多胺"),
    # ---- 有机相酰氯 ----
    ("TMC",        "O=C(Cl)C1=CC(C(=O)Cl)=CC(C(=O)Cl)=C1",      "均苯三甲酰氯（经典）"),
    ("IPC",        "O=C(Cl)c1cccc(C(=O)Cl)c1",                  "间苯二甲酰氯"),
    ("TPC",        "O=C(Cl)c1ccc(C(=O)Cl)cc1",                  "对苯二甲酰氯"),
    ("BTEC",       "O=C(Cl)c1cc(C(=O)Cl)c(C(=O)Cl)cc1C(=O)Cl",  "均苯四甲酰氯"),
    ("ADC",        "O=C(Cl)CCCCC(=O)Cl",                        "己二酰氯（脂肪族）"),
    ("SEC",        "O=C(Cl)CCCCCCCC(=O)Cl",                     "癸二酰氯"),
    ("CHDC",       "O=C(Cl)C1CCC(C(=O)Cl)CC1",                  "环己烷二甲酰氯"),
    ("BPC",        "O=C(Cl)c1ccc(-c2ccc(C(=O)Cl)cc2)cc1",       "联苯二甲酰氯"),
]

_dataset = set()
for _c in ("smiles_aq", "smiles_org"):
    _dataset |= {v for v in OPTIONS.get(_c, []) if _clean_smiles(v)}

# 名称映射：库内单体显示中文名，数据集内未收录的显示 SMILES 片段
MONOMER_NAMES = {sm: f"{nm}｜{desc}" for nm, sm, desc in EXTERNAL_MONOMERS}

_lib = {sm for _, sm, _ in EXTERNAL_MONOMERS if _clean_smiles(sm)}
_all = _dataset | _lib
AQ_CANDIDATES  = sorted([v for v in _all if _is_amine(v)])
ORG_CANDIDATES = sorted([v for v in _all if _is_acyl(v)])
IN_DATASET     = _dataset   # 标记来源，供界面区分

app = FastAPI(title="NF Mg/Li 分离性能预测平台", version="1.0")


# ------------------ 请求体 ------------------
class MembraneInput(BaseModel):
    Foundational_Membrane: str = "Psf"
    SMILES_part1: str = ""
    Density_part1: float = 0.5          # wt%
    Covered_Time_1: float = 120         # s
    SMILES_part2: str = ""
    Density_part2: float = 0.15
    Covered_Time_2: float = 60
    WCA: Optional[float] = None
    Zeta_Potential: Optional[float] = None
    Average_pore_diameter: Optional[float] = None
    MWCO: Optional[float] = None
    Thickness: Optional[float] = None
    Pressure: float = 5
    Feed: float = 1000
    two_stage: bool = False    # True=用第一阶段预测表征补齐空缺（正向设计）


class ScreenInput(BaseModel):
    base: MembraneInput
    D1_list: List[float] = [0.2, 0.5, 1.0, 1.5, 2.0]
    D2_list: List[float] = [0.10, 0.15, 0.20, 0.30]
    T1_list: List[float] = [60, 120, 300]
    T2_list: List[float] = [30, 60, 120]
    top_n: int = 15


class MonomerScreenInput(BaseModel):
    Foundational_Membrane: str = "Psf"
    # 表征上下文（留空则按中位数填补，预测上限偏低；默认给高性能膜典型表征）
    WCA: Optional[float] = 48
    Zeta_Potential: Optional[float] = 14
    Average_pore_diameter: Optional[float] = 0.40
    MWCO: Optional[float] = 360
    Thickness: Optional[float] = 81
    Pressure: float = 10
    Feed: float = 2000
    # 每个单体对扫描的合成条件网格
    D1_list: List[float] = [0.5, 1.0, 2.0]
    D2_list: List[float] = [0.15, 0.30]
    T2_list: List[float] = [60]
    T1_list: List[float] = [120, 300]
    T2: float = 60
    top_n: int = 20
    two_stage: bool = True     # 默认开：每个单体用第一阶段预测其表征，使筛选自洽


# ------------------ 工具函数 ------------------
def _input_to_frame(inp: MembraneInput) -> pd.DataFrame:
    row = {c: np.nan for c in (SMILES_COLS + DENS_COLS + TIME_COLS +
                               CHAR_COLS + OP_COLS + [MEMBRANE_COL])}
    row[MEMBRANE_COL] = inp.Foundational_Membrane
    row["SMILES_part1"] = inp.SMILES_part1
    row["SMILES_part2"] = inp.SMILES_part2
    row["Density_part1(wt%)"] = inp.Density_part1
    row["Density_part2"] = inp.Density_part2
    row["Covered Time_1(s)"] = inp.Covered_Time_1
    row["Covered Time_2(s)"] = inp.Covered_Time_2
    row["WCA(°)"] = inp.WCA
    row["Zeta Potential"] = inp.Zeta_Potential
    row["Average pore diameter"] = inp.Average_pore_diameter
    row["MWCO(Da)"] = inp.MWCO
    row["Thickness(nm)"] = inp.Thickness
    row["Pressure(bar)"] = inp.Pressure
    row["Feed(ppm)"] = inp.Feed
    return pd.DataFrame([row])


def predict_characterization(frame):
    """第一阶段：由配方预测膜表征。返回 {列名: 预测值数组}。
    旧模型（无 stage1）时返回空 dict。"""
    if not STAGE1_MODELS or STAGE1_FEATURES is None:
        return {}
    Xr = recipe_only_features(frame.copy(), MEMBRANE_CATS)
    Xr = Xr.reindex(columns=STAGE1_FEATURES).values
    out = {}
    for col, mdl in STAGE1_MODELS.items():
        z = mdl.predict(Xr)
        out[col] = stage1_inv(col, z)
    return out


def _fill_predicted_char(frame):
    """将 frame 中缺失的膜表征用第一阶段预测值补齐（两阶段前向）。
    仅填补用户未提供的表征列，用户显式输入的值优先保留。"""
    pred = predict_characterization(frame)
    if not pred:
        return frame, {}
    used = {}
    for col, vals in pred.items():
        if col not in frame.columns:
            frame[col] = vals; used[col] = True; continue
        cur = pd.to_numeric(frame[col], errors="coerce").values
        filled = cur.copy().astype(float)
        miss = np.isnan(filled)
        filled[miss] = np.asarray(vals)[miss]
        frame[col] = filled
        used[col] = bool(miss.any())
    return frame, used


def _predict_core(inp: MembraneInput):
    frame = _input_to_frame(inp)
    char_source = "user"
    if getattr(inp, "two_stage", False) and STAGE1_MODELS:
        frame, used = _fill_predicted_char(frame)
        if any(used.values()):
            char_source = "stage1"
    X = build_feature_matrix(frame, feature_names=FEATURE_NAMES,
                             membrane_categories=MEMBRANE_CATS).values
    out = {}
    for key in TARGETS:
        mb = MODELS[key]
        z = mb["model"].predict(X)[0]
        zl = mb["q_lo"].predict(X)[0]
        zh = mb["q_hi"].predict(X)[0]
        p = float(inv_transform(key, z))
        lo, hi = sorted([float(inv_transform(key, zl)), float(inv_transform(key, zh))])
        if key in ("Mg_rej", "Li_rej"):
            p = float(np.clip(p, 0, 100)); lo = float(np.clip(lo, 0, 100)); hi = float(np.clip(hi, 0, 100))
        out[key] = {"value": round(p, 2), "lo": round(lo, 2), "hi": round(hi, 2)}

    score, comp = composite_score(
        out["Mg_rej"]["value"], out["Li_rej"]["value"],
        flux_mg=out["Flux_Mg"]["value"], pwp=out["PWP"]["value"],
        return_components=True)
    return {
        "predictions": out,
        "composite_score": round(float(np.atleast_1d(score)[0]), 4),
        "separation_balance": round(float(np.atleast_1d(comp["separation"])[0]), 4),
        "mg_quality": round(float(np.atleast_1d(comp["mg_quality"])[0]), 4),
        "li_quality": round(float(np.atleast_1d(comp["li_quality"])[0]), 4),
        "char_source": char_source,
    }


# ------------------ 接口 ------------------
@app.get("/api/options")
def get_options():
    return OPTIONS

@app.get("/api/health")
def health():
    """Lightweight verification endpoint for local and cloud deployments."""
    return {
        "status": "ok",
        "service": "Nanofiltration Intelligence Platform",
        "two_stage_available": bool(STAGE1_MODELS),
        "targets": list(TARGETS),
        "n_membrane_options": len(OPTIONS.get("membranes", [])),
    }

@app.get("/api/metrics")
def get_metrics():
    return METRICS

@app.get("/api/importance")
def get_importance():
    return IMPORTANCES

@app.post("/api/predict")
def predict(inp: MembraneInput):
    try:
        return _predict_core(inp)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/screen")
def screen(req: ScreenInput):
    b = req.base
    rows, meta = [], []
    for d1 in req.D1_list:
        for d2 in req.D2_list:
            for t1 in req.T1_list:
                for t2 in req.T2_list:
                    rows.append(_row_from(
                        b.Foundational_Membrane, b.SMILES_part1, b.SMILES_part2,
                        d1, t1, d2, t2, b.WCA, b.Zeta_Potential,
                        b.Average_pore_diameter, b.MWCO, b.Thickness,
                        b.Pressure, b.Feed))
                    meta.append((d1, d2, t1, t2))
    if len(rows) > 20000:
        raise HTTPException(status_code=400,
            detail=f"组合数过多（{len(rows)}），请减少各列表的取值个数（建议总数 ≤ 20000）")
    preds, scores = _predict_batch(rows)
    out = []
    for i, (d1, d2, t1, t2) in enumerate(meta):
        out.append({"D1": d1, "D2": d2, "T1": t1, "T2": t2,
                    "Mg_rej": round(float(preds["Mg_rej"][i]), 2),
                    "Li_rej": round(float(preds["Li_rej"][i]), 2),
                    "Flux_Mg": round(float(preds["Flux_Mg"][i]), 2),
                    "score": round(float(scores[i]), 4)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return {"results": out[:req.top_n], "n_total": len(out)}


def _predict_batch(rows):
    """批量预测：一次构造特征矩阵、模型一次推理，比逐条循环快 1~2 个数量级。
    rows: list[dict]，每个 dict 为一行原始输入（列名同数据集）。
    返回 (predictions_dict_of_arrays, scores_array)"""
    df = pd.DataFrame(rows)
    X = build_feature_matrix(df, feature_names=FEATURE_NAMES,
                             membrane_categories=MEMBRANE_CATS).values
    out = {}
    for k in TARGETS:
        z = MODELS[k]["model"].predict(X)
        v = inv_transform(k, z)
        if k in ("Mg_rej", "Li_rej"):
            v = np.clip(v, 0, 100)
        out[k] = v
    scores = composite_score(out["Mg_rej"], out["Li_rej"],
                             flux_mg=out["Flux_Mg"], pwp=out["PWP"])
    return out, np.asarray(scores)


def _row_from(membrane, s1, s2, d1, t1, d2, t2, wca, zeta, pore, mwco, thick, press, feed):
    """构造一行原始输入（与训练列名一致）。"""
    row = {c: np.nan for c in (SMILES_COLS + DENS_COLS + TIME_COLS +
                               CHAR_COLS + OP_COLS + [MEMBRANE_COL])}
    row[MEMBRANE_COL] = membrane
    row["SMILES_part1"] = s1
    row["SMILES_part2"] = s2
    row["Density_part1(wt%)"] = d1
    row["Covered Time_1(s)"] = t1
    row["Density_part2"] = d2
    row["Covered Time_2(s)"] = t2
    row["WCA(°)"] = wca
    row["Zeta Potential"] = zeta
    row["Average pore diameter"] = pore
    row["MWCO(Da)"] = mwco
    row["Thickness(nm)"] = thick
    row["Pressure(bar)"] = press
    row["Feed(ppm)"] = feed
    return row


@app.get("/api/monomers")
def get_monomers():
    """返回清洗后的候选单体：按官能团分流（含 N 胺→水相，含酰氯→有机相）。"""
    return {"aq": AQ_CANDIDATES, "org": ORG_CANDIDATES,
            "names": MONOMER_NAMES,
            "n_aq": len(AQ_CANDIDATES), "n_org": len(ORG_CANDIDATES),
            "n_from_dataset": len(_dataset), "n_from_library": len(_lib)}


@app.get("/api/stage1")
def get_stage1():
    """返回第一阶段（配方→表征）各表征的预测精度；旧模型返回空。"""
    return {"available": bool(STAGE1_MODELS), "report": STAGE1_REPORT,
            "chars": list(STAGE1_MODELS.keys())}


@app.get("/api/shap")
def get_shap():
    """返回各目标的 SHAP 特征贡献摘要（训练时生成 models/shap.json）。"""
    path = os.path.join(MODELS_DIR, "shap.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404,
            detail="尚无 SHAP 数据，请重新训练一次以生成 models/shap.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/screen_monomers")
def screen_monomers(req: MonomerScreenInput):
    """潜力单体筛选：遍历所有 (水相胺 × 有机相酰氯) 组合，
    对每个组合扫描合成条件网格取其最优得分（即该单体对的潜力上限）。
    采用批量向量化预测，全部组合一次推理完成。"""
    rows, meta = [], []
    for s1 in AQ_CANDIDATES:
        for s2 in ORG_CANDIDATES:
            for d1 in req.D1_list:
                for d2 in req.D2_list:
                    for t1 in req.T1_list:
                        for t2 in req.T2_list:
                            rows.append(_row_from(
                                req.Foundational_Membrane, s1, s2, d1, t1, d2, t2,
                                req.WCA, req.Zeta_Potential, req.Average_pore_diameter,
                                req.MWCO, req.Thickness, req.Pressure, req.Feed))
                            meta.append((s1, s2, d1, d2, t1, t2))

    # 两阶段：为每个单体组合预测其自身表征，替换固定上下文（使筛选自洽）
    two_stage_used = False
    if getattr(req, "two_stage", True) and STAGE1_MODELS:
        fr = pd.DataFrame(rows)
        pred_char = predict_characterization(fr)
        if pred_char:
            for col, vals in pred_char.items():
                fr[col] = vals
            rows = fr.to_dict("records")
            two_stage_used = True
    preds, scores = _predict_batch(rows)

    # 每个单体对取最优条件
    best = {}
    for idx, (s1, s2, d1, d2, t1, t2) in enumerate(meta):
        key = (s1, s2)
        if key not in best or scores[idx] > best[key]["score"]:
            best[key] = {
                "aq": s1, "org": s2,
                "aq_name": MONOMER_NAMES.get(s1, ""), "org_name": MONOMER_NAMES.get(s2, ""),
                "aq_in_dataset": bool(s1 in IN_DATASET), "org_in_dataset": bool(s2 in IN_DATASET),
                "Mg_rej": round(float(preds["Mg_rej"][idx]), 2),
                "Li_rej": round(float(preds["Li_rej"][idx]), 2),
                "Flux_Mg": round(float(preds["Flux_Mg"][idx]), 2),
                "PWP": round(float(preds["PWP"][idx]), 2),
                "score": round(float(scores[idx]), 4),
                "best_D1": d1, "best_D2": d2, "best_T1": t1, "best_T2": t2,
            }
    results = sorted(best.values(), key=lambda x: x["score"], reverse=True)

    by_aq = {}
    for r in results:
        if r["aq"] not in by_aq or r["score"] > by_aq[r["aq"]]["score"]:
            by_aq[r["aq"]] = r
    aq_rank = sorted(by_aq.values(), key=lambda x: x["score"], reverse=True)
    return {"results": results[:req.top_n],
            "aq_ranking": [{"aq": r["aq"], "aq_name": r.get("aq_name",""),
                            "in_dataset": r.get("aq_in_dataset", False),
                            "best_score": r["score"],
                            "best_mg": r["Mg_rej"]} for r in aq_rank[:15]],
            "n_pairs": len(results),
            "n_combinations": len(rows),
            "two_stage": two_stage_used,
            "n_aq": len(AQ_CANDIDATES), "n_org": len(ORG_CANDIDATES)}


# ---- 静态文件 & 首页 ----
STATIC_DIR = os.path.join(BASE, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()
