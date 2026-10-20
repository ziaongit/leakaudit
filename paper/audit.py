"""Paper 2 reframe: explanation-based label-leakage audit.
Runs on dataset A (z-score label) and dataset B (externally supplied label)."""
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import lightgbm as lgb, xgboost as xgb, shap
import prep, prep_ext

OUT = 'results/audit.json'; R = {}
def models(seed, pos_w):
    return {
        'LGBM': lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1, scale_pos_weight=pos_w),
        'XGB': xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=seed, scale_pos_weight=pos_w, eval_metric='logloss', n_jobs=4),
        'RF': RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=5, class_weight='balanced', random_state=seed, n_jobs=4),
    }
def best_f1(y, p):
    ts = np.linspace(0.05, 0.95, 19); return max(f1_score(y, p >= t) for t in ts)
def glob_shap(m, Xs):
    sv = shap.TreeExplainer(m).shap_values(Xs)
    if isinstance(sv, list): sv = sv[1]
    if sv.ndim == 3: sv = sv[:, :, 1]
    return np.abs(sv).mean(0), sv
def jacc(a, b, k):
    a, b = set(a[:k]), set(b[:k]); return len(a & b) / len(a | b)

def audit(name, X, y, rule_feats, derived, seeds=(0, 1, 2, 3, 4), n_shap=2000):
    feats = list(X.columns); pos_w = (1 - y.mean()) / y.mean(); res = {'n': len(X), 'prev': float(y.mean()), 'features': feats, 'runs': []}
    imps = {}
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=s)
        Xs = Xte.sample(min(n_shap, len(Xte)), random_state=s)
        for mn, m in models(s, pos_w).items():
            m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
            imp, _ = glob_shap(m, Xs if mn != 'RF' else Xs.iloc[:500]); imp = imp / imp.sum()
            order = [feats[i] for i in np.argsort(-imp)]
            share_rule = float(sum(imp[feats.index(f)] for f in rule_feats if f in feats))
            share_der = float(sum(imp[feats.index(f)] for f in derived if f in feats))
            run = dict(seed=s, model=mn, f1=best_f1(yte, p), auc=roc_auc_score(yte, p), ap=average_precision_score(yte, p),
                       top1=order[0], top1_share=float(imp.max()), share_rule=share_rule, share_derived=share_der,
                       top5=order[:5], imp=dict(zip(feats, map(float, imp))))
            res['runs'].append(run); imps[(s, mn)] = imp
            print(name, s, mn, f"F1={run['f1']:.3f} AUC={run['auc']:.3f} top1={order[0]} {imp.max():.2f} rule={share_rule:.2f} der={share_der:.2f}", flush=True)
    # stability: across seeds within model, and across models within seed
    def pair_stats(pairs):
        rho, tau, j5 = [], [], []
        for a, b in pairs:
            ia, ib = imps[a], imps[b]; rho.append(spearmanr(ia, ib)[0]); tau.append(kendalltau(ia, ib)[0])
            oa = list(np.argsort(-ia)); ob = list(np.argsort(-ib)); j5.append(jacc(oa, ob, 5))
        return dict(spearman=float(np.mean(rho)), kendall=float(np.mean(tau)), jaccard5=float(np.mean(j5)), n=len(pairs))
    ms = ['LGBM', 'XGB', 'RF']
    res['stab_seed'] = {m: pair_stats([((a, m), (b, m)) for i, a in enumerate(seeds) for b in seeds[i + 1:]]) for m in ms}
    res['stab_model'] = pair_stats([((s, a), (s, b)) for s in seeds for i, a in enumerate(ms) for b in ms[i + 1:]])
    return res

def surrogate(X, y, depths=(1, 2, 3, 4)):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=0, verbose=-1, scale_pos_weight=(1 - y.mean()) / y.mean()).fit(Xtr, ytr)
    ptr = (m.predict_proba(Xtr)[:, 1] >= 0.5).astype(int); pte = (m.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
    out = {}
    for d in depths:
        t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ptr)
        out[d] = dict(fidelity=float((t.predict(Xte) == pte).mean()), f1_vs_label=float(f1_score(yte, t.predict(Xte))),
                      used=sorted(set(X.columns[i] for i in t.tree_.feature if i >= 0)))
        if d == 3: out['tree3'] = export_text(t, feature_names=list(X.columns), max_depth=3)
    return out

def lime_stability(X, y, n_inst=30, reps=5, k=5):
    sys.path.insert(0, '/home/claude/vendor/lime-0.2.0.1'); from lime.lime_tabular import LimeTabularExplainer
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=0, verbose=-1, scale_pos_weight=(1 - y.mean()) / y.mean()).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    idx = list(np.argsort(-p)[:n_inst // 2]) + list(np.random.RandomState(0).choice(len(Xte), n_inst // 2, replace=False))
    J, top1_rule = [], []
    for i in idx:
        tops = []
        for r in range(reps):
            e = LimeTabularExplainer(Xtr.values, feature_names=list(X.columns), class_names=['ok', 'fraud'], discretize_continuous=True, random_state=r)
            ex = e.explain_instance(Xte.values[i], m.predict_proba, num_features=k, num_samples=2000)
            tops.append([X.columns[f] for f, _ in ex.as_map()[1]])
        J.append(np.mean([jacc(tops[a], tops[b], k) for a in range(reps) for b in range(a + 1, reps)])); top1_rule.append(tops[0][0])
    vc = pd.Series(top1_rule).value_counts(normalize=True)
    return dict(jaccard5_mean=float(np.mean(J)), jaccard5_min=float(np.min(J)), top1_freq={k_: float(v) for k_, v in vc.head(5).items()}, n=len(idx), reps=reps)

if __name__ == '__main__':
    which = sys.argv[1]
    df, ch = prep.load(verbose=False); df = df.drop(columns=['index'], errors='ignore')
    yA = df.pop('Fraud'); XA = df
    RULE = [ch, 'Provider Type']; DER = ['PCR', 'Charge_Allowed_Ratio']
    if which == 'A':
        R['A_full'] = audit('A_full', XA, yA, RULE, DER)
    elif which == 'Anocharge':
        Xn = XA.drop(columns=[ch] + DER)
        R['A_nocharge'] = audit('A_nocharge', Xn, yA, ['Provider Type'], [], seeds=(0, 1, 2, 3, 4))
    elif which == 'B':
        B = prep_ext.load_ext(verbose=False); yB = B.pop('Fraud')
        R['B'] = audit('B', B, yB, [], [], seeds=(0, 1, 2, 3, 4), n_shap=1082)
    elif which == 'rule':
        Xr = XA[[ch, 'Provider Type']]
        R['A_ruleonly'] = audit('A_ruleonly', Xr, yA, RULE, [], seeds=(0, 1, 2, 3, 4))
        R['surrogate_A'] = surrogate(XA, yA)
        B = prep_ext.load_ext(verbose=False); yB = B.pop('Fraud'); R['surrogate_B'] = surrogate(B, yB)
    elif which == 'rule5':
        R['A_ruleonly'] = audit('A_ruleonly', XA[[ch, 'Provider Type']], yA, RULE, [])
    elif which == 'lime':
        R['lime_A'] = lime_stability(XA, yA)
        B = prep_ext.load_ext(verbose=False); yB = B.pop('Fraud'); R['lime_B'] = lime_stability(B, yB)
    json.dump(R, open(f'results/audit_{which}.json', 'w'), indent=1, default=str)
