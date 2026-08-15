"""Strict, group-aware audit of Mg2+ rejection models.

This is an experiment script: every reported prediction is generated for a
recipe group that was absent from the corresponding training fold.  It is
intended to separate an honest generalisation score from optimistic row-wise
splits and to test whether a classifier--regressor cascade helps.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from app.nf_features import (TARGETS, MEMBRANE_COL, DENS_COLS, TIME_COLS, CHAR_COLS,
                             OP_COLS, build_feature_matrix, build_synth_key,
                             fwd_transform, inv_transform)

try:
    from lightgbm import LGBMRegressor
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

RANDOM_STATE = 42


def load_data():
    data_dir = os.path.join(ROOT, "data")
    paths = [os.path.join(data_dir, p) for p in os.listdir(data_dir)
             if p.endswith(".xlsx") and not p.startswith("~$")]
    frames = []
    for p in sorted(paths):
        xl = pd.ExcelFile(p)
        sheet = "main" if "main" in xl.sheet_names else xl.sheet_names[0]
        frames.append(pd.read_excel(p, sheet_name=sheet))
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["synth_key"] = build_synth_key(df)
    df = df.drop_duplicates(subset=["synth_key"] + OP_COLS + list(TARGETS.values()), keep="first")
    for c in DENS_COLS + TIME_COLS + CHAR_COLS + OP_COLS + list(TARGETS.values()):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def et(leaf=1, max_features=0.7, indicators=False):
    return Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=indicators)),
                     ("m", ExtraTreesRegressor(n_estimators=700, min_samples_leaf=leaf,
                                                 max_features=max_features, random_state=RANDOM_STATE,
                                                 n_jobs=1))])


def rf(leaf=2):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", RandomForestRegressor(n_estimators=700, min_samples_leaf=leaf,
                                                   max_features=0.7, random_state=RANDOM_STATE,
                                                   n_jobs=1))])


def gbr(depth=2, leaf=4):
    from sklearn.ensemble import GradientBoostingRegressor
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", GradientBoostingRegressor(n_estimators=350, learning_rate=0.025,
                                                       max_depth=depth, min_samples_leaf=leaf,
                                                       loss="huber", random_state=RANDOM_STATE))])


def hgb(leaf=10):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", HistGradientBoostingRegressor(max_iter=350, learning_rate=0.04,
                                                          max_leaf_nodes=15, min_samples_leaf=leaf,
                                                          l2_regularization=2.0,
                                                          random_state=RANDOM_STATE))])


def svr(c=3.0):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("m", SVR(C=c, epsilon=1.5, gamma="scale"))])


def ridge():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("m", Ridge(alpha=20.0))])


def lgbm():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", LGBMRegressor(n_estimators=250, num_leaves=7, learning_rate=0.025,
                                           min_child_samples=18, reg_lambda=8.0, reg_alpha=0.2,
                                           colsample_bytree=0.75, random_state=RANDOM_STATE,
                                           n_jobs=1, verbosity=-1))])


def risk_classifier():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", ExtraTreesClassifier(n_estimators=700, max_features=0.7,
                                                 min_samples_leaf=3, class_weight="balanced",
                                                 random_state=RANDOM_STATE, n_jobs=1))])


def pred_with_transform(model, xtr, ytr, xte, transform):
    yt = transform[0](ytr)
    return transform[1](clone(model).fit(xtr, yt).predict(xte))


RAW = (lambda y: y, lambda y: y)
COMP = (lambda y: fwd_transform("Mg_rej", y), lambda y: inv_transform("Mg_rej", y))
LOGIT = (lambda y: np.log(np.clip(y, 0.5, 99.5) / np.clip(100-y, 0.5, 99.5)),
         lambda z: 100 / (1 + np.exp(-np.clip(z, -12, 12))))


def group_oof(model, X, y, groups, transform):
    pred = np.zeros(len(y))
    cv = GroupKFold(n_splits=5)
    for tr, te in cv.split(X, y, groups):
        pred[te] = pred_with_transform(model, X[tr], y[tr], X[te], transform)
    return pred


def two_stage_oof(reg_model, X, y, groups, threshold, transform, soft=True):
    """Classifier -> specialised regressors, evaluated without group leakage."""
    pred = np.zeros(len(y)); prob = np.zeros(len(y))
    cv = GroupKFold(n_splits=5)
    for tr, te in cv.split(X, y, groups):
        ytr, yte = y[tr], y[te]
        fail = (ytr < threshold).astype(int)
        clf = risk_classifier().fit(X[tr], fail)
        p = clf.predict_proba(X[te])[:, 1]
        prob[te] = p
        # If a fold has too few failures, a dedicated tail regressor is unstable;
        # fall back to the all-data regressor instead of fabricating an improvement.
        all_pred = pred_with_transform(reg_model, X[tr], ytr, X[te], transform)
        if fail.sum() < 14 or (len(fail) - fail.sum()) < 20:
            pred[te] = all_pred
            continue
        low_pred = pred_with_transform(reg_model, X[tr][fail == 1], ytr[fail == 1], X[te], transform)
        high_pred = pred_with_transform(reg_model, X[tr][fail == 0], ytr[fail == 0], X[te], transform)
        if soft:
            pred[te] = p * low_pred + (1 - p) * high_pred
        else:
            pred[te] = np.where(p >= 0.5, low_pred, high_pred)
    return pred, prob


def metric_row(name, y, pred, kind, extra=None):
    out = {"method": name, "setting": kind,
           "GroupKFold_R2": round(float(r2_score(y, pred)), 4),
           "GroupKFold_MAE": round(float(mean_absolute_error(y, pred)), 4),
           "bias": round(float(np.mean(pred-y)), 4),
           "tail_MAE_y_lt_90": round(float(mean_absolute_error(y[y < 90], pred[y < 90])), 4)}
    if extra:
        out.update(extra)
    return out


def split_gap(model, X, y, groups):
    """Compare row-wise and recipe-held-out evaluation with the same estimator."""
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    row_pred = np.zeros(len(y))
    for tr, te in kf.split(X):
        row_pred[te] = pred_with_transform(model, X[tr], y[tr], X[te], COMP)
    scores = []
    gss = GroupShuffleSplit(n_splits=20, test_size=0.2, random_state=RANDOM_STATE)
    for tr, te in gss.split(X, y, groups):
        p = pred_with_transform(model, X[tr], y[tr], X[te], COMP)
        scores.append(r2_score(y[te], p))
    return {"rowwise_kfold_R2": round(float(r2_score(y, row_pred)), 4),
            "group_shuffle_R2_mean": round(float(np.mean(scores)), 4),
            "group_shuffle_R2_sd": round(float(np.std(scores)), 4),
            "group_shuffle_R2_min": round(float(np.min(scores)), 4),
            "group_shuffle_R2_max": round(float(np.max(scores)), 4)}


def main():
    df = load_data()
    cats = sorted(df[MEMBRANE_COL].fillna("Unk").astype(str).unique().tolist())
    all_features = build_feature_matrix(df.copy(), membrane_categories=cats)
    # A compact physics subset is an ablation: it tells whether high-dimensional
    # recipe descriptors are helping or simply adding variance on n=321.
    physics_cols = [c for c in all_features.columns if c.startswith(("WCA", "Zeta", "Ra", "Rq", "Average", "MWCO", "Thickness", "Modification", "zeta_", "steric_", "donnan", "pore_", "mwco_", "pressure", "feed", "p_", "log_feed"))]
    m = df[TARGETS["Mg_rej"]].notna()
    X = all_features.loc[m].values
    Xphysical = all_features.loc[m, physics_cols].values
    y = df.loc[m, TARGETS["Mg_rej"]].values.astype(float)
    g = df.loc[m, "synth_key"].values

    experiments = [("ET leaf=1 mf=0.3", et(1, .3), X),
                   ("ET leaf=1 mf=0.5", et(1, .5), X),
                   ("ET leaf=1 mf=0.7", et(1, .7), X),
                   ("ET leaf=1 mf=1.0", et(1, 1.0), X),
                   ("ET leaf=1 mf=0.7 + missing", et(1, .7, True), X),
                   ("ET leaf=2 mf=0.5", et(2, .5), X),
                   ("ET leaf=2 mf=0.7", et(2, .7), X),
                   ("ET leaf=3 mf=0.7", et(3, .7), X), ("ET leaf=4 mf=0.7", et(4, .7), X),
                   ("ET leaf=2 physical", et(2, 1.0), Xphysical),
                   ("RF leaf=1", rf(1), X), ("RF leaf=2", rf(2), X),
                   ("GBR depth=2", gbr(2, 4), X), ("GBR depth=3", gbr(3, 5), X),
                   ("HGB leaf=10", hgb(10), X), ("HGB leaf=18", hgb(18), X),
                   ("SVR C=3", svr(3.0), X), ("SVR C=10", svr(10.0), X), ("Ridge", ridge(), X)]
    if HAVE_LGBM:
        experiments.append(("LightGBM regularized", lgbm(), X))

    rows = []
    preds = {}
    for name, model, xx in experiments:
        # Tree models are also assessed under the legacy transform; non-tree
        # baselines are evaluated on raw scale to keep the audit compact.
        transforms = [("raw", RAW), ("complement_log", COMP), ("logit", LOGIT)] if name.startswith("ET") else [("raw", RAW)]
        for tname, tf in transforms:
            p = group_oof(model, xx, y, g, tf)
            rows.append(metric_row(name, y, p, tname))
            preds[(name, tname)] = (p, model, xx, tf)
    rows.sort(key=lambda z: (-z["GroupKFold_R2"], z["GroupKFold_MAE"]))

    # Averaging independently-randomised forests is a low-variance ensemble;
    # all components use exactly the same held-out recipe folds, so these OOF
    # predictions remain leakage-free (the blend weights are fixed, not fitted).
    blend_specs = [
        ("ET raw blend 0.3+0.7", [("ET leaf=1 mf=0.3", "raw"), ("ET leaf=1 mf=0.7", "raw")]),
        ("ET raw blend 0.3+0.5+0.7+1.0", [("ET leaf=1 mf=0.3", "raw"), ("ET leaf=1 mf=0.5", "raw"),
                                            ("ET leaf=1 mf=0.7", "raw"), ("ET leaf=1 mf=1.0", "raw")]),
    ]
    blends = []
    for label, keys in blend_specs:
        bp = np.mean([preds[k][0] for k in keys], axis=0)
        blends.append(metric_row(label, y, bp, "fixed_equal_weight"))
    rows.extend(blends)
    rows.sort(key=lambda z: (-z["GroupKFold_R2"], z["GroupKFold_MAE"]))

    cascades = []
    # Test the best single regressor as the base to avoid making an arbitrary
    # comparison that disadvantages the cascade.
    # The cascade uses the strongest fitted single regressor, not a precomputed
    # blend, because its component needs to be re-fitted inside each outer fold.
    best_name, best_tf_name = max((k for k in preds), key=lambda k: r2_score(y, preds[k][0]))
    p0, best_model, best_x, best_tf = preds[(best_name, best_tf_name)]
    for threshold in (80.0, 85.0, 90.0):
        for soft in (True, False):
            p, risk_p = two_stage_oof(best_model, best_x, y, g, threshold, best_tf, soft=soft)
            lab = (y < threshold).astype(int)
            info = {"threshold": threshold, "gate": "soft" if soft else "hard",
                    "failure_n": int(lab.sum()),
                    "classifier_PR_AUC": round(float(average_precision_score(lab, risk_p)), 4),
                    "classifier_ROC_AUC": round(float(roc_auc_score(lab, risk_p)), 4)}
            cascades.append(metric_row("classifier+regressor", y, p, "two_stage", info))
    cascades.sort(key=lambda z: (-z["GroupKFold_R2"], z["GroupKFold_MAE"]))

    report = {
        "protocol": "All primary scores are out-of-fold predictions under GroupKFold grouped by synthesis recipe. No recipe appears in both fitting and evaluation folds.",
        "n_rows": int(len(y)), "n_recipe_groups": int(pd.Series(g).nunique()),
        "target_summary": {"mean": round(float(y.mean()), 3), "sd": round(float(y.std()), 3),
                           "min": round(float(y.min()), 3), "max": round(float(y.max()), 3),
                           "n_lt80": int((y < 80).sum()), "n_lt85": int((y < 85).sum()), "n_lt90": int((y < 90).sum())},
        "evaluation_gap": split_gap(et(2, .7), X, y, g),
        "single_stage": rows,
        "two_stage": cascades,
        "conclusion": "Choose a cascade only if its group-held-out R2 exceeds the best single-stage model. A strong classifier alone is a useful risk warning, not evidence that continuous regression improved."
    }
    out = os.path.join(ROOT, "results", "mg_model_audit.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
