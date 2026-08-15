"""Repeated random-split check for the small apparent SVR improvement."""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from app.nf_features import (TARGETS, MEMBRANE_COL, DENS_COLS, TIME_COLS, CHAR_COLS, OP_COLS, build_feature_matrix, build_synth_key, fwd_transform, inv_transform)
def et(mf):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("m", ExtraTreesRegressor(n_estimators=600, max_features=mf, random_state=42, n_jobs=1))])
def svr():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("m", SVR(C=10, epsilon=.08, gamma="scale"))])
p = os.path.join(ROOT, "_recovery_package", "nf_webapp", "data", "nanofiltration_data.xlsx")
df = pd.read_excel(p, sheet_name="main")
df["synth_key"] = build_synth_key(df)
df = df.drop_duplicates(subset=["synth_key"] + OP_COLS + list(TARGETS.values()), keep="first")
for c in DENS_COLS + TIME_COLS + CHAR_COLS + OP_COLS + list(TARGETS.values()):
    df[c] = pd.to_numeric(df[c], errors="coerce")
cats = sorted(df[MEMBRANE_COL].fillna("Unk").astype(str).unique())
m = df[TARGETS["Mg_rej"]].notna()
X = build_feature_matrix(df.copy(), membrane_categories=cats).loc[m].values
yraw = df.loc[m, TARGETS["Mg_rej"]].values
y = fwd_transform("Mg_rej", yraw)
def predict(seed, kind):
    cv = KFold(n_splits=5, shuffle=True, random_state=seed); p = np.zeros(len(y))
    for tr, te in cv.split(X):
        if kind == "ET_blend":
            p[te] = .5*et(.5).fit(X[tr], y[tr]).predict(X[te]) + .5*et(.7).fit(X[tr], y[tr]).predict(X[te])
        else:
            p[te] = svr().fit(X[tr], y[tr]).predict(X[te])
    raw = inv_transform("Mg_rej", p)
    return r2_score(y, p), r2_score(yraw, raw), mean_absolute_error(yraw, raw)
report = {}
seeds = list(range(10)) + [42]
for kind in ("ET_blend", "SVR_C10"):
    vals = np.array([predict(s, kind) for s in seeds])
    report[kind] = {"R2_transformed_mean": round(float(vals[:10,0].mean()),4), "R2_transformed_sd": round(float(vals[:10,0].std()),4), "R2_raw_mean": round(float(vals[:10,1].mean()),4), "R2_raw_sd": round(float(vals[:10,1].std()),4), "MAE_raw_mean": round(float(vals[:10,2].mean()),4), "seed42": [round(float(x),4) for x in vals[-1]]}
out = {"protocol":"10 repeated 5-fold random KFold seeds (0–9), plus the original fixed seed 42.", "results":report}
with open(os.path.join(ROOT,"results","mg_candidate_stability.json"),"w",encoding="utf-8") as f:
    json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2))
