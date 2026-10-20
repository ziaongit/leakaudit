"""leakaudit: explanation-based label-leakage audit for tabular classifiers.

Five steps (see the paper):
  1. Attribution concentration (SHAP share of each feature)
  2. Suspect-only model (decisive test)
  3. Removal and proxy check
  4. Global explanation stability across splits and models
  5. Local check with repeated LIME explanations
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def default_models(pos_weight: float, seed: int) -> dict:
    """LightGBM, XGBoost and Random Forest with class weighting (each only if installed)."""
    ms = {}
    try:
        import lightgbm as lgb
        ms["LightGBM"] = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=seed,
                                            verbose=-1, scale_pos_weight=pos_weight)
    except ImportError:
        pass
    try:
        import xgboost as xgb
        ms["XGBoost"] = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=seed,
                                          scale_pos_weight=pos_weight, eval_metric="logloss", n_jobs=4)
    except ImportError:
        pass
    ms["RandomForest"] = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=5,
                                                class_weight="balanced", random_state=seed, n_jobs=4)
    return ms


def best_f1(y, p, grid=np.linspace(0.05, 0.95, 19)) -> float:
    """F1 at the best threshold on a grid (optimistic; use only for comparisons)."""
    return float(max(f1_score(y, p >= t) for t in grid))


def shap_share(model, X: pd.DataFrame) -> pd.Series:
    """Step 1: each feature's share of mean |SHAP| (sums to 1)."""
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    imp = np.abs(sv).mean(0)
    return pd.Series(imp / imp.sum(), index=X.columns).sort_values(ascending=False)


def _fit_eval(model, Xtr, ytr, Xte, yte):
    m = clone(model).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return m, best_f1(yte, p), float(roc_auc_score(yte, p))


def _jaccard(a, b, k):
    a, b = set(a[:k]), set(b[:k])
    return len(a & b) / len(a | b)


def audit(X: pd.DataFrame, y, suspects: list[str] | None = None, derived: list[str] | None = None,
          seeds=(0, 1, 2, 3, 4), models: dict | None = None, n_shap: int = 2000, test_size: float = 0.2,
          lime_records: int = 0, verbose: bool = True) -> dict:
    """Run Steps 1-4 (and Step 5 if lime_records > 0).

    suspects: features believed to define the label. If None, the top-ranked SHAP feature from
              Step 1 is used.
    derived:  features computed from the suspects; removed together with them in Step 3.
    models:   dict name -> unfitted sklearn-style classifier. Default: LightGBM, XGBoost, RF.
    """
    y = pd.Series(np.asarray(y), index=X.index)
    pos_w = float((1 - y.mean()) / y.mean())
    rep = {"n": len(X), "prevalence": float(y.mean()), "runs": []}
    imps = {}
    # ---- Step 1 (+ full-model performance) ----
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=s)
        Xs = Xte.sample(min(n_shap, len(Xte)), random_state=s)
        for name, mdl in (models or default_models(pos_w, s)).items():
            m, f1, auc = _fit_eval(mdl, Xtr, ytr, Xte, yte)
            share = shap_share(m, Xs if name != "RandomForest" else Xs.iloc[:500])
            imps[(s, name)] = share.reindex(X.columns).values
            rep["runs"].append(dict(step="full", seed=s, model=name, f1=f1, auc=auc,
                                    top1=share.index[0], top1_share=float(share.iloc[0])))
    full = pd.DataFrame(rep["runs"])
    mean_share = pd.DataFrame(imps, index=X.columns).T.mean().sort_values(ascending=False)
    rep["mean_share"] = mean_share.to_dict()
    if suspects is None:
        suspects = [mean_share.index[0]]
    derived = derived or []
    rep["suspects"], rep["derived"] = list(suspects), list(derived)
    rep["suspect_share"] = float(mean_share[list(suspects)].sum())

    # ---- Steps 2 and 3 ----
    keep = [c for c in X.columns if c not in set(suspects) | set(derived)]
    rows = []
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=s)
        for name, mdl in (models or default_models(pos_w, s)).items():
            _, f1o, auco = _fit_eval(mdl, Xtr[list(suspects)], ytr, Xte[list(suspects)], yte)
            mr, f1r, aucr = _fit_eval(mdl, Xtr[keep], ytr, Xte[keep], yte)
            sh = shap_share(mr, Xte[keep].sample(min(n_shap if name != "RandomForest" else 500, len(Xte)), random_state=s))
            rows.append(dict(seed=s, model=name, f1_suspect_only=f1o, auc_suspect_only=auco,
                             f1_removed=f1r, auc_removed=aucr, new_top1=sh.index[0], new_top1_share=float(sh.iloc[0])))
    st = pd.DataFrame(rows)
    summary = full.groupby("model")[["f1", "auc"]].mean().join(
        st.groupby("model")[["f1_suspect_only", "auc_suspect_only", "f1_removed", "auc_removed", "new_top1_share"]].mean())
    summary["new_top1"] = st.groupby("model").new_top1.agg(lambda v: v.value_counts().index[0])
    rep["summary"] = summary.round(3)
    rep["steps23"] = st

    # ---- Step 4: stability ----
    names = sorted({k[1] for k in imps})
    def stats(pairs):
        r = [spearmanr(imps[a], imps[b])[0] for a, b in pairs]
        t = [kendalltau(imps[a], imps[b])[0] for a, b in pairs]
        j = [_jaccard(list(np.argsort(-imps[a])), list(np.argsort(-imps[b])), 5) for a, b in pairs]
        return dict(spearman=float(np.mean(r)), kendall=float(np.mean(t)), jaccard5=float(np.mean(j)))
    rep["stability_across_splits"] = {n: stats([((a, n), (b, n)) for i, a in enumerate(seeds) for b in seeds[i + 1:]])
                                      for n in names} if len(seeds) > 1 else {}
    rep["stability_across_models"] = stats([((s, a), (s, b)) for s in seeds for i, a in enumerate(names)
                                            for b in names[i + 1:]]) if len(names) > 1 else {}

    # ---- Step 5: LIME ----
    if lime_records:
        rep["lime"] = lime_check(X, y, n_records=lime_records, seed=seeds[0])

    rep["verdict"] = verdict(rep)
    if verbose:
        print_report(rep)
    return rep


def lime_check(X, y, n_records=30, repeats=5, k=5, seed=0, model=None) -> dict:
    """Step 5: repeat LIME on the same records; report top-1 agreement and top-k Jaccard."""
    from lime.lime_tabular import LimeTabularExplainer
    y = pd.Series(np.asarray(y), index=X.index)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    pos_w = float((1 - y.mean()) / y.mean())
    m = clone(model or list(default_models(pos_w, seed).values())[0]).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    idx = list(np.argsort(-p)[: n_records // 2]) + list(
        np.random.RandomState(seed).choice(len(Xte), n_records - n_records // 2, replace=False))
    J, top1 = [], []
    for i in idx:
        tops = []
        for r in range(repeats):
            e = LimeTabularExplainer(Xtr.values, feature_names=list(X.columns), discretize_continuous=True, random_state=r)
            ex = e.explain_instance(Xte.values[i], m.predict_proba, num_features=k, num_samples=2000)
            tops.append([X.columns[f] for f, _ in ex.as_map()[1]])
        J.append(np.mean([_jaccard(tops[a], tops[b], k) for a in range(repeats) for b in range(a + 1, repeats)]))
        top1.append(tops[0][0])
    vc = pd.Series(top1).value_counts(normalize=True)
    return dict(top1=vc.index[0], top1_freq=float(vc.iloc[0]), jaccard_mean=float(np.mean(J)), jaccard_min=float(np.min(J)))


def verdict(rep: dict, tol: float = 0.02, drop: float = 0.15) -> str:
    """Plain-language verdict from Steps 2-3 (the decisive tests)."""
    s = rep["summary"]
    if (s["auc"] < 0.6).all():
        return "INCONCLUSIVE: the full model is close to chance (ROC-AUC < 0.6), so there is little signal to audit."
    matches = (s["f1_suspect_only"] >= s["f1"] - tol).all()
    big_drop = ((s["f1"] - s["f1_removed"]) >= drop).all()
    if matches and big_drop:
        return ("LIKELY LEAKAGE: the suspect features alone match the full model, and removing them "
                "causes a large drop. Check how the label was built.")
    if matches:
        return ("POSSIBLE LEAKAGE: the suspect features alone match the full model, but other features "
                "carry the same information. Inspect correlated features.")
    return "NO SIGN OF LEAKAGE from these suspects: the suspect features alone do worse than the full model."


def print_report(rep: dict) -> None:
    print(f"\nleakaudit report  (n={rep['n']}, prevalence={rep['prevalence']:.3%})")
    print("Step 1  top features by mean SHAP share:")
    for f, v in list(rep["mean_share"].items())[:5]:
        print(f"        {v:6.3f}  {f}")
    print(f"        suspects {rep['suspects']} hold {rep['suspect_share']:.1%} of attribution")
    print("Steps 2-3 (F1 at best threshold):")
    print(rep["summary"].to_string())
    if rep.get("stability_across_models"):
        print("Step 4  stability across models:", {k: round(v, 3) for k, v in rep["stability_across_models"].items()})
    if rep.get("lime"):
        print("Step 5  LIME:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep["lime"].items()})
    print("Verdict:", rep["verdict"])
