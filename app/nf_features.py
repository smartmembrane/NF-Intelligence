"""
nf_features.py —— 共享特征工程模块
训练脚本与 FastAPI 服务共用，确保线上线下特征完全一致。
不依赖 RDKit；若已安装 RDKit，可在 smiles_descriptors 中替换为真实描述符。
"""
import numpy as np
import pandas as pd

SMILES_COLS = [f"SMILES_part{i}" for i in range(1, 6)]
DENS_COLS = ["Density_part1(wt%)", "Density_part2", "Density_part3",
             "Density_part4", "Density_part5"]
TIME_COLS = [f"Covered Time_{i}(s)" for i in range(1, 6)]
CHAR_COLS = ["WCA(°)", "Zeta Potential", "Ra(nm)", "Rq(nm)",
             "Average pore diameter", "MWCO(Da)", "Thickness(nm)",
             "Modification Temperature(℃)", "Modification Time(s)"]
OP_COLS = ["Pressure(bar)", "Feed(ppm)"]
MEMBRANE_COL = "Foundational Membrane"

TARGETS = {
    "PWP": "PWP", "Mg_rej": "Mg2+", "Flux_Mg": "Flux Mg2+",
    "Li_rej": "Li+", "Flux_Li": "Flux Li+",
}

# 目标变换：截留率左偏 -> log1p(100-y)；通量/PWP 右偏 -> log1p(y)
def fwd_transform(name, y):
    y = np.asarray(y, float)
    if name in ("Mg_rej", "Li_rej"):
        return np.log1p(np.clip(100 - y, 0, None))
    return np.log1p(np.clip(y, 0, None))

def inv_transform(name, z):
    z = np.asarray(z, float)
    if name in ("Mg_rej", "Li_rej"):
        return 100 - np.expm1(z)
    return np.expm1(z)


# RDKit 可选支持：本机 pip install rdkit-pypi 后自动启用真实分子描述符（准确度更高）
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors as _RD
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False


import re as _re

def parse_monomer(entry):
    """解析单体记录。支持聚合物记法 {SMILES}tag（tag 为分子量/规格标签，如 PEI 不同分子量）。
    返回 (基础SMILES, mw_tag)；非聚合物记法时 mw_tag=None。"""
    if not isinstance(entry, str):
        return entry, None
    m = _re.match(r"^\{(.+)\}\s*([\d.]+)\s*$", entry.strip())
    if m:
        try:
            return m.group(1), float(m.group(2))
        except ValueError:
            return m.group(1), None
    return entry, None


def smiles_descriptors(smi):
    """SMILES 描述符：优先用 RDKit 真实计算，未安装则退回字符级近似。
    自动剥离聚合物记法 {SMILES}tag，用基础SMILES计算（tag 由 structure_features 单独入模）。"""
    smi, _ = parse_monomer(smi)
    if RDKIT_AVAILABLE and isinstance(smi, str) and smi.strip() not in ("", "nan"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return dict(
                mw=_RD.MolWt(mol),
                nN=sum(1 for a in mol.GetAtoms() if a.GetSymbol()=="N"),
                nO=sum(1 for a in mol.GetAtoms() if a.GetSymbol()=="O"),
                nCl=sum(1 for a in mol.GetAtoms() if a.GetSymbol()=="Cl"),
                nF=sum(1 for a in mol.GetAtoms() if a.GetSymbol()=="F"),
                nC=sum(1 for a in mol.GetAtoms() if a.GetSymbol()=="C"),
                n_ring=mol.GetRingInfo().NumRings(),
                n_amine=_RD.NumHDonors(mol),
                n_acyl=_RD.NumHAcceptors(mol),
                n_arom=sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),
                length=_RD.TPSA(mol))   # length 槽位复用为 TPSA（极性表面积）
    # ---- 退回字符级近似 ----
    if not isinstance(smi, str) or smi.strip() in ("", "nan"):
        return dict(mw=0.0, nN=0, nO=0, nCl=0, nF=0, nC=0,
                    n_ring=0, n_amine=0, n_acyl=0, n_arom=0, length=0)
    nN = smi.count("N") + smi.count("n")
    nO = smi.count("O") + smi.count("o")
    nCl = smi.count("Cl")
    nF = smi.count("F")
    nC = smi.count("C") + smi.count("c") - nCl
    return dict(
        mw=nC*12 + nN*14 + nO*16 + nCl*35.5 + nF*19,
        nN=nN, nO=nO, nCl=nCl, nF=nF, nC=nC,
        n_ring=sum(c.isdigit() for c in smi) // 2,
        n_amine=nN,
        n_acyl=smi.count("C(=O)Cl") + smi.count("C(=O)O"),
        n_arom=smi.count("c") + smi.count("n") + smi.count("o"),
        length=len(smi))

DESC_KEYS = list(smiles_descriptors("C").keys())


def aggregate_molecular(frame):
    out = {f"agg_{k}": np.zeros(len(frame)) for k in DESC_KEYS}
    out.update({f"wtd_{k}": np.zeros(len(frame)) for k in DESC_KEYS})
    dens = frame[DENS_COLS].fillna(0).values
    for j, scol in enumerate(SMILES_COLS):
        d = frame[scol].apply(smiles_descriptors)
        w = dens[:, j]
        for k in DESC_KEYS:
            v = d.apply(lambda x: x[k]).values.astype(float)
            out[f"agg_{k}"] += v
            out[f"wtd_{k}"] += v * w
    return pd.DataFrame(out, index=frame.index)


def synthesis_features(frame, mol):
    o = pd.DataFrame(index=frame.index)
    D1 = frame["Density_part1(wt%)"].fillna(0)
    D2 = frame["Density_part2"].fillna(0)
    T1 = frame["Covered Time_1(s)"].fillna(0)
    T2 = frame["Covered Time_2(s)"].fillna(0)
    o["D1_aq"], o["D2_org"], o["T1_rxn"], o["T2_rxn"] = D1, D2, T1, T2
    o["conc_x_time"] = D1 * T1
    o["sqrt_t1"] = np.sqrt(T1.clip(lower=0))
    o["sqrt_t2"] = np.sqrt(T2.clip(lower=0))
    o["d_ratio"] = D1 / (D2 + 1e-6)
    o["d1_sq"] = D1 ** 2
    o["d1_x_d2"] = D1 * D2
    o["ip_balance"] = ((D1.values * np.maximum(mol["agg_n_amine"].values, 0)) /
                       (D2.values * np.maximum(mol["agg_nCl"].values, 0) + 1e-6)).clip(0, 50)
    return o


def structure_features(frame):
    """多步反应/聚合物结构特征：
    - mw_tag_j：各单体位的聚合物分子量标签（如 PEI 3.4/19/132k），非聚合物为 NaN
    - wtd_mw_tag：浓度加权的分子量标签合计（聚合物总投入规格）
    - n_steps：非空单体位数（反应步数代理）
    - n_polymer：聚合物单体位数"""
    o = pd.DataFrame(index=frame.index)
    dens = frame[DENS_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).values
    wtd = np.zeros(len(frame)); npoly = np.zeros(len(frame)); nsteps = np.zeros(len(frame))
    for j, col in enumerate(SMILES_COLS):
        tags = pd.to_numeric(frame[col].apply(lambda v: parse_monomer(v)[1]), errors="coerce")
        o[f"mw_tag_{j+1}"] = tags.astype(float)
        filled = frame[col].notna() & frame[col].astype(str).str.strip().ne("")
        nsteps += filled.values.astype(float)
        has_tag = tags.notna().values
        npoly += has_tag.astype(float)
        wtd += tags.fillna(0).astype(float).values * dens[:, j]
    o["wtd_mw_tag"] = wtd
    o["n_steps"] = nsteps
    o["n_polymer"] = npoly
    return o


def operating_features(frame):
    o = pd.DataFrame(index=frame.index)
    P = frame["Pressure(bar)"]
    Fe = frame["Feed(ppm)"]
    P = P.fillna(P.median() if P.notna().any() else 5.0)
    Fe = Fe.fillna(Fe.median() if Fe.notna().any() else 1000.0)
    o["pressure"] = P
    o["feed"] = Fe
    o["p_sq"] = P ** 2
    o["p_x_feed"] = P * Fe
    o["log_feed"] = np.log1p(Fe)
    return o


def char_features(frame):
    ch = frame[CHAR_COLS].copy()
    ch.columns = [c.replace("(°)", "_d").replace("(nm)", "_nm")
                   .replace("(Da)", "_Da").replace("(℃)", "_C")
                   .replace("(s)", "_s").replace(" ", "_") for c in ch.columns]
    return ch


def physics_features(frame):
    """NF 传质物理特征（Donnan 排斥 + 空间位阻代理）。
    水合离子半径：Mg2+ = 0.428 nm，Li+ = 0.382 nm。缺失表征时给 NaN，由填补器处理。"""
    o = pd.DataFrame(index=frame.index)
    zeta = pd.to_numeric(frame["Zeta Potential"], errors="coerce")
    pore = pd.to_numeric(frame["Average pore diameter"], errors="coerce")
    mwco = pd.to_numeric(frame["MWCO(Da)"], errors="coerce")
    thick = pd.to_numeric(frame["Thickness(nm)"], errors="coerce")
    R_MG, R_LI = 0.428, 0.382
    o["zeta_sq"] = zeta ** 2                      # Donnan 排斥强度 ∝ ζ²
    o["zeta_abs"] = zeta.abs()
    o["steric_Mg"] = pore / (2 * R_MG)            # 孔径/水合直径，<1 强位阻
    o["steric_Li"] = pore / (2 * R_LI)
    o["steric_gap"] = o["steric_Li"] - o["steric_Mg"]   # 两离子位阻差 → 选择性来源
    o["donnan_proxy"] = zeta ** 2 / (pore ** 2 + 1e-6)  # Donnan-位阻耦合
    o["pore_sq"] = pore ** 2
    o["mwco_per_thk"] = mwco / (thick + 1e-6)     # 传质阻力代理
    o["log_mwco"] = np.log1p(mwco.clip(lower=0))
    return o


def build_feature_matrix(frame, feature_names=None, membrane_categories=None):
    """组装完整特征矩阵。
    feature_names: 训练时保存的列顺序；预测时传入以对齐。
    membrane_categories: 训练集出现过的基底膜类别列表。
    """
    for c in DENS_COLS + TIME_COLS + CHAR_COLS + OP_COLS:
        if c in frame:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")

    mol = aggregate_molecular(frame)
    syn = synthesis_features(frame, mol)
    op = operating_features(frame)
    ch = char_features(frame)

    feat = pd.concat([ch, physics_features(frame), structure_features(frame), syn, op, mol], axis=1)
    feat = feat.loc[:, ~feat.columns.duplicated()]

    # 基底膜 one-hot
    memb_series = frame[MEMBRANE_COL].fillna("Unk") if MEMBRANE_COL in frame else pd.Series(["Unk"] * len(frame))
    if membrane_categories is not None:
        for cat in membrane_categories:
            col = f"base_{cat}"
            feat[col] = (memb_series == cat).astype(float).values
    else:
        dummies = pd.get_dummies(memb_series, prefix="base").astype(float)
        feat = pd.concat([feat, dummies], axis=1)

    if feature_names is not None:
        feat = feat.reindex(columns=feature_names)
    return feat


# ============ 两阶段建模：第一阶段（配方 → 表征）============
# 第一阶段要预测的膜表征（第二阶段会用到的关键量）；部分做 log 变换
STAGE1_CHARS = {
    "Zeta Potential": None,
    "Average pore diameter": None,
    "MWCO(Da)": "log",
    "WCA(°)": None,
    "Thickness(nm)": None,
}


def recipe_only_features(frame, membrane_categories):
    """仅用配方信息（分子结构 + 合成条件 + 操作条件 + 基底膜）构造特征，
    严格不含任何膜表征列 —— 用于第一阶段「配方 → 表征」预测，避免循环依赖。"""
    for c in DENS_COLS + TIME_COLS + OP_COLS:
        if c in frame:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    mol = aggregate_molecular(frame)
    syn = synthesis_features(frame, mol)
    op = operating_features(frame)
    feat = pd.concat([syn, op, mol], axis=1)
    feat = feat.loc[:, ~feat.columns.duplicated()]
    memb = frame[MEMBRANE_COL].fillna("Unk") if MEMBRANE_COL in frame else pd.Series(["Unk"] * len(frame))
    for cat in membrane_categories:
        feat[f"base_{cat}"] = (memb == cat).astype(float).values
    return feat


def stage1_fwd(col, y):
    import numpy as _np
    return _np.log1p(_np.clip(y, 0, None)) if STAGE1_CHARS.get(col) == "log" else y


def stage1_inv(col, z):
    import numpy as _np
    return _np.expm1(z) if STAGE1_CHARS.get(col) == "log" else z


def build_synth_key(frame):
    key_cols = SMILES_COLS + DENS_COLS + TIME_COLS
    return frame[key_cols].apply(lambda r: "|".join(map(str, r.values)), axis=1)


class EnsembleModel:
    """对若干已拟合子模型在变换空间加权平均。定义在共享模块以便 joblib 反序列化。"""
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights
    def predict(self, X):
        import numpy as _np
        w = getattr(self, "weights", None)
        if not w:
            w = [1.0/len(self.models)]*len(self.models)
        return _np.sum([wi*m.predict(X) for wi, m in zip(w, self.models)], axis=0)


# ------------------ 综合评价指标（镁锂平衡·调和平均，不使用 SF、不使用差分、无硬阈值）------------------
def mg_retention_logscore(r_mg, cap=99.95):
    """Mg 截留对数项：-log10(1-R/100)。95%→1.30，99%→2.00，99.9%→3.00。"""
    r = np.clip(np.atleast_1d(np.asarray(r_mg, float)), 0, cap)
    return -np.log10(1 - r / 100)


def composite_score(r_mg, r_li, flux_mg=None, pwp=None,
                    w_sep=0.80, w_flux=0.12, w_pwp=0.08,
                    flux_ref=20.0, pwp_ref=15.0,
                    return_components=False, **_ignored):
    """
    镁锂平衡分离得分（0~1，越高越好）：

      分离项 = 调和平均( Mg截留质量, Li透过质量 )
        · Mg截留质量 = -log10(1-R_Mg/100)/3，截断到[0,1]（高截留区灵敏）
        · Li透过质量 = 1 - R_Li/100（Li 截留越低越好）
        · 调和平均的性质：两项都高才得高分，任何一项接近 0 整体趋于 0
          —— 用数学结构本身实现"镁锂平衡"，无需硬阈值，也不依赖差分 Δrej。

      总分 = 0.80×分离项 + 0.12×通量项 + 0.08×PWP项
    """
    r_mg = np.atleast_1d(np.asarray(r_mg, float))
    r_li = np.atleast_1d(np.asarray(r_li, float))
    mg_q = np.clip(mg_retention_logscore(r_mg) / 3.0, 0, 1)   # Mg 截留质量
    li_q = 1 - np.clip(r_li, 0, 100) / 100                     # Li 透过质量
    sep = 2 * mg_q * li_q / (mg_q + li_q + 1e-9)               # 调和平均

    flux_term = (np.clip(np.atleast_1d(np.asarray(flux_mg, float)) / flux_ref, 0, 1)
                 if flux_mg is not None else np.zeros_like(r_mg))
    if flux_mg is None:
        w_flux = 0.0
    pwp_term = (np.clip(np.atleast_1d(np.asarray(pwp, float)) / pwp_ref, 0, 1)
                if pwp is not None else np.zeros_like(r_mg))
    if pwp is None:
        w_pwp = 0.0
    s = w_sep + w_flux + w_pwp
    score = (w_sep*sep + w_flux*flux_term + w_pwp*pwp_term) / s
    if return_components:
        return score, dict(mg_quality=mg_q, li_quality=li_q, separation=sep,
                           flux_norm=flux_term, pwp_norm=pwp_term)
    return score
