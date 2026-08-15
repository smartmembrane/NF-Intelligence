"""Non-destructive Mg2+ model benchmark.

This script never writes models/ or changes the web app.  It evaluates each
candidate with the original random KFold protocol *and* a recipe-held-out
GroupKFold protocol, reporting both transformed-space and raw-percent R2.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                              HistGradientBoostingRegressor, GradientBoostingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.svm import SVR

try:
    from catboost import CatBoostRegressor
    HAVE_CATBOOST = True
except Exception:
    HAVE_CATBOOST = False
try:
    from xgboost import XGBRegressor
    HAVE_XGBOOST = True
except Exception:
    HAVE_XGBOOST = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from app.nf_features import (DENS_COLS, TIME_COLS, CHAR_COLS, OP_COLS, MEMBRANE_COL,
                             SMILES_COLS, TARGETS, build_feature_matrix, build_synth_key,
                             fwd_transform, inv_transform)

RANDOM_STATE = 42


def data_path():
    # Use the verified original copy when an open Excel process locks data/.
    p = os.path.join(ROOT, "data", "nanofiltration_data.xlsx")
    if os.path.exists(p):
        try:
            pd.ExcelFile(p).close()
            return p
        except Exception:
            pass
    return os.path.join(ROOT, "_recovery_package", "nf_webapp", "data", "nanofiltration_data.xlsx")


def load():
    p = data_path()
    xl = pd.ExcelFile(p)
    df = pd.read_excel(p, sheet_name="main" if "main" in xl.sheet_names else xl.sheet_names[0])
    df["synth_key"] = build_synth_key(df)
    df = df.drop_duplicates(subset=["synth_key"] + OP_COLS + list(TARGETS.values()), keep="first").reset_index(drop=True)
    for c in DENS_COLS + TIME_COLS + CHAR_COLS + OP_COLS + list(TARGETS.values()):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def et(max_features=.7, leaf=1):
    return ExtraTreesRegressor(n_estimators=600, max_features=max_features,
                               min_samples_leaf=leaf, random_state=RANDOM_STATE, n_jobs=1)


def numeric_pipe(model):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("m", model)])


def selected_et(k, max_features=.5):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("select", SelectKBest(mutual_info_regression, k=k)),
                     ("m", et(max_features))])


def svr(c):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("m", SVR(C=c, epsilon=.08, gamma="scale"))])


def gbr(depth=2, leaf=4):
    return GradientBoostingRegressor(n_estimators=450, learning_rate=.025, max_depth=depth,
                                     min_samples_leaf=leaf, loss="huber", random_state=RANDOM_STATE)


def hgb(leaf=12):
    return HistGradientBoostingRegressor(max_iter=450, learning_rate=.04, max_leaf_nodes=15,
                                         min_samples_leaf=leaf, l2_regularization=1.5,
                                         random_state=RANDOM_STATE)


def catboost():
    return CatBoostRegressor(iterations=500, depth=5, learning_rate=.025, l2_leaf_reg=8,
                             loss_function="RMSE", random_seed=RANDOM_STATE, verbose=False,
                             allow_writing_files=False, thread_count=1)


def hybrid_pipe(numeric_cols, categorical_cols, max_features=.7, leaf=1):
    # Exact monomer identity is added only within the folds. Unknown molecules
    # are ignored at prediction time, so this is an interpolation enhancement,
    # not a claim of improved new-chemistry extrapolation.
    prep = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric_cols),
        ("monomer", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="__missing__")),
                               ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
    ])
    return Pipeline([("prep", prep), ("m", et(max_features=max_features, leaf=leaf))])


def score(name, model, X, y_raw, groups, transformed=True):
    y = fwd_transform("Mg_rej", y_raw) if transformed else y_raw
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gkf = GroupKFold(n_splits=5)
    pk = cross_val_predict(model, X, y, cv=kf, n_jobs=1)
    pg = cross_val_predict(model, X, y, cv=gkf, groups=groups, n_jobs=1)
    raw_k = inv_transform("Mg_rej", pk) if transformed else pk
    raw_g = inv_transform("Mg_rej", pg) if transformed else pg
    return {
        "method": name,
        "target_scale": "log(100-R)" if transformed else "raw_percent",
        "KFold_R2_transformed": round(float(r2_score(y, pk)), 4) if transformed else None,
        "KFold_R2_raw": round(float(r2_score(y_raw, raw_k)), 4),
        "KFold_MAE_raw": round(float(mean_absolute_error(y_raw, raw_k)), 4),
        "GroupKFold_R2_raw": round(float(r2_score(y_raw, raw_g)), 4),
        "GroupKFold_MAE_raw": round(float(mean_absolute_error(y_raw, raw_g)), 4),
    }


def main():
    df = load()
    cats = sorted(df[MEMBRANE_COL].fillna("Unk").astype(str).unique().tolist())
    numerical = build_feature_matrix(df.copy(), membrane_categories=cats)
    mask = df[TARGETS["Mg_rej"]].notna()
    Xn = numerical.loc[mask].copy()
    # SMILES is retained as categorical identity in addition to descriptors.
    Xh = Xn.copy()
    for c in SMILES_COLS:
        Xh[c] = df.loc[mask, c].fillna("__missing__").astype(str).values
    y = df.loc[mask, TARGETS["Mg_rej"]].values.astype(float)
    g = df.loc[mask, "synth_key"].values
    numeric_cols = Xn.columns.tolist()
    cat_cols = SMILES_COLS

    specs = [
        ("Original-style ET 0.5", numeric_pipe(et(.5)), Xn, True),
        ("Original-style ET 0.7", numeric_pipe(et(.7)), Xn, True),
        ("Original-style ET 1.0", numeric_pipe(et(1.0)), Xn, True),
        ("ET raw 0.7", numeric_pipe(et(.7)), Xn, False),
        ("ET + SMILES one-hot 0.3", hybrid_pipe(numeric_cols, cat_cols, .3), Xh, True),
        ("ET + SMILES one-hot 0.5", hybrid_pipe(numeric_cols, cat_cols, .5), Xh, True),
        ("ET + SMILES one-hot 0.7", hybrid_pipe(numeric_cols, cat_cols, .7), Xh, True),
        ("ET + SMILES one-hot 1.0", hybrid_pipe(numeric_cols, cat_cols, 1.0), Xh, True),
        ("RandomForest 0.5", numeric_pipe(RandomForestRegressor(n_estimators=600, max_features=.5,
                                                                  min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=1)), Xn, True),
        ("GradientBoosting depth=2", numeric_pipe(gbr(2, 4)), Xn, True),
        ("GradientBoosting depth=3", numeric_pipe(gbr(3, 5)), Xn, True),
        ("HistGradientBoosting", numeric_pipe(hgb(12)), Xn, True),
        ("ET + MI top-20", selected_et(20, .5), Xn, True),
        ("ET + MI top-35", selected_et(35, .5), Xn, True),
        ("ET + MI top-50", selected_et(50, .5), Xn, True),
        ("SVR C=3", svr(3), Xn, True),
        ("SVR C=10", svr(10), Xn, True),
    ]
    if HAVE_CATBOOST:
        specs.append(("CatBoost", numeric_pipe(catboost()), Xn, True))
    if HAVE_XGBOOST:
        specs += [
            ("XGBoost shallow", numeric_pipe(XGBRegressor(n_estimators=450, max_depth=2, learning_rate=.025,
                                                             min_child_weight=5, subsample=.8, colsample_bytree=.7,
                                                             reg_lambda=8, objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=1, verbosity=0)), Xn, True),
            ("XGBoost depth=3", numeric_pipe(XGBRegressor(n_estimators=450, max_depth=3, learning_rate=.02,
                                                            min_child_weight=7, subsample=.8, colsample_bytree=.65,
                                                            reg_lambda=12, objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=1, verbosity=0)), Xn, True),
        ]
    rows = [score(name, model, xx, y, g, transformed=transformed)
            for name, model, xx, transformed in specs]
    rows.sort(key=lambda z: (-z["KFold_R2_raw"], z["KFold_MAE_raw"]))
    out = {
        "protocol": "5-fold random KFold(seed=42) plus recipe-held-out GroupKFold. All metrics are reported on raw Mg2+ percent; transformed R2 is included only for comparison with the original UI.",
        "n": int(len(y)), "n_groups": int(pd.Series(g).nunique()), "results": rows,
        "interpretation": "A SMILES one-hot gain is valid for interpolation among seen monomers but should not be claimed as improved extrapolation to unseen monomer chemistry."
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "mg_interpolation_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
