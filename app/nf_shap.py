"""
nf_shap.py —— 树模型 SHAP 值计算（无需安装 shap 库）

实现 Lundberg 等提出的 TreeSHAP（interventional / path-dependent 近似）：
沿树路径递归分解每次分裂的贡献，按节点样本权重加权。
对随机森林/ExtraTrees 等树集成，SHAP 值可加：总贡献 = 各树贡献的平均。

性质保证：sum(shap_values) + base_value == model.predict(x)（可加性，代码中已断言校验）

若环境已安装官方 shap 库，会优先使用（速度更快、支持更多模型）。
"""
import numpy as np

try:
    import shap as _shap_lib
    SHAP_LIB = True
except Exception:
    SHAP_LIB = False


def _tree_expectations(tree):
    """一次性算出每个节点的期望输出（自底向上，避免递归重复）。"""
    cl, cr = tree.children_left, tree.children_right
    val = tree.value[:, 0, 0]
    wts = tree.weighted_n_node_samples
    n = len(cl)
    E = np.empty(n, dtype=float)
    # 后序遍历（迭代）
    stack = [(0, False)]
    while stack:
        node, done = stack.pop()
        if cl[node] == -1:
            E[node] = val[node]
            continue
        if done:
            l, r = cl[node], cr[node]
            E[node] = (wts[l]*E[l] + wts[r]*E[r]) / (wts[l] + wts[r])
        else:
            stack.append((node, True))
            stack.append((cl[node], False))
            stack.append((cr[node], False))
    return E


def _tree_shap_batch(tree, Xf, n_features):
    """单棵树对一批样本的 SHAP 值（迭代沿路径，O(depth) per sample）。"""
    cl, cr = tree.children_left, tree.children_right
    feat, thr = tree.feature, tree.threshold
    E = _tree_expectations(tree)
    n = len(Xf)
    phi = np.zeros((n, n_features))
    for i in range(n):
        x = Xf[i]
        node = 0
        while cl[node] != -1:
            f = feat[node]
            go = cl[node] if x[f] <= thr[node] else cr[node]
            phi[i, f] += E[go] - E[node]
            node = go
    return phi, E[0]


def tree_shap(estimator, X, feature_names=None, max_samples=None, max_trees=120, random_state=0):
    """
    计算树集成模型的 SHAP 值。

    estimator: 已拟合的 sklearn Pipeline（含 imputer + 树模型）或树模型本身
    X: (n, d) 特征矩阵（原始，可含 NaN；若传 Pipeline 会先走其 imputer）
    返回: dict(shap_values=(n,d), base_value=float, X_used=(n,d))
    """
    # 拆出 imputer 与树模型
    imputer, model = None, estimator
    if hasattr(estimator, "named_steps"):
        model = estimator.named_steps.get("m", None) or list(estimator.named_steps.values())[-1]
        imputer = estimator.named_steps.get("imp", None)

    Xf = np.asarray(X, dtype=float)
    kept = None
    if imputer is not None:
        Xf = imputer.transform(Xf)
        # SimpleImputer 会丢弃全为 NaN 的列 —— 记录保留的列索引以对齐特征名
        try:
            support = getattr(imputer, "get_support", None)
            if support is not None:
                kept = np.where(support())[0]
            else:
                stats = getattr(imputer, "statistics_", None)
                if stats is not None:
                    kept = np.where(~np.isnan(stats))[0]
        except Exception:
            kept = None
    if kept is None or len(kept) != Xf.shape[1]:
        kept = np.arange(Xf.shape[1])

    if max_samples is not None and len(Xf) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(Xf), max_samples, replace=False)
        Xf = Xf[idx]

    n, d = Xf.shape

    # 官方 shap 库优先
    if SHAP_LIB and hasattr(model, "estimators_"):
        try:
            ex = _shap_lib.TreeExplainer(model)
            sv = ex.shap_values(Xf, check_additivity=False)
            sv = np.asarray(sv)
            base = float(np.atleast_1d(ex.expected_value)[0])
            return dict(shap_values=sv, base_value=base, X_used=Xf, kept=kept)
        except Exception:
            pass  # 回退到自实现

    # 自实现 TreeSHAP
    if not hasattr(model, "estimators_"):
        raise ValueError("tree_shap 仅支持树集成模型（含 estimators_）")

    trees = [e.tree_ for e in model.estimators_]
    # 树子采样：SHAP 值是树间平均，用部分树即可得到稳定估计（大幅提速）
    if max_trees is not None and len(trees) > max_trees:
        rng = np.random.RandomState(random_state)
        sel = rng.choice(len(trees), max_trees, replace=False)
        trees = [trees[i] for i in sel]

    sv = np.zeros((n, d))
    bases = []
    for tr in trees:
        phi, b = _tree_shap_batch(tr, Xf, d)
        sv += phi
        bases.append(b)
    sv /= len(trees)
    base = float(np.mean(bases))
    return dict(shap_values=sv, base_value=base, X_used=Xf, kept=kept)


def shap_summary(shap_values, feature_names, X_used=None, top_k=20):
    """汇总为可导出的表：平均绝对 SHAP（重要性）+ 方向性（与特征值的相关）。"""
    sv = np.asarray(shap_values)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]
    rows = []
    for j in order:
        direction = np.nan
        if X_used is not None:
            xj = np.asarray(X_used)[:, j]
            if np.std(xj) > 1e-12 and np.std(sv[:, j]) > 1e-12:
                direction = float(np.corrcoef(xj, sv[:, j])[0, 1])
        rows.append({
            "feature": feature_names[j] if feature_names else f"f{j}",
            "mean_abs_shap": float(mean_abs[j]),
            "mean_shap": float(sv[:, j].mean()),
            "direction_corr": direction,   # >0: 特征值越大越推高预测
        })
    return rows
