"""Robustness checks for Paper 2 (reviewer-proofing).
mode 'perf'   : all feature sets x 3 models x 5 splits; F1 at a threshold tuned on a validation split
                taken from the training data (no test-set tuning), plus the old test-best F1, AUC, AP.
mode 'shapall': dataset A with ALL suspects removed (charge, provider type, PCR, Charge_Allowed_Ratio): SHAP top-1.
mode 'group'  : dataset A, provider-grouped splits (same NPI never in train and test), LightGBM.
"""
import os, sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.base import clone
import lightgbm as lgb, xgboost as xgb, shap
from sklearn.ensemble import RandomForestClassifier
import prep, prep_ext

GRID = np.linspace(0.05, 0.95, 19)
def models(seed, pw):
    return {'LGBM': lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1, scale_pos_weight=pw),
            'XGB': xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=seed, scale_pos_weight=pw, eval_metric='logloss', n_jobs=4),
            'RF': RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=5, class_weight='balanced', random_state=seed, n_jobs=4)}
def evaluate(m, Xtr, ytr, Xte, yte, seed):
    # tune threshold on a validation split of the training data, then refit on all training data
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=100 + seed)
    pv = clone(m).fit(Xa, ya).predict_proba(Xv)[:, 1]
    thr = GRID[int(np.argmax([f1_score(yv, pv >= t) for t in GRID]))]
    mf = clone(m).fit(Xtr, ytr); p = mf.predict_proba(Xte)[:, 1]
    return mf, dict(f1_val=f1_score(yte, p >= thr), thr=float(thr), f1_best=max(f1_score(yte, p >= t) for t in GRID),
                    auc=roc_auc_score(yte, p), ap=average_precision_score(yte, p))

def loadA():
    df, ch = prep.load(verbose=False); df = df.drop(columns=['index'], errors='ignore'); y = df.pop('Fraud'); return df, y, ch
def loadB():
    B = prep_ext.load_ext(verbose=False); y = B.pop('Fraud'); return B, y

mode = sys.argv[1]; out = []
if mode == 'perf':
    which = sys.argv[2]
    if which == 'A':
        X, y, ch = loadA()
        sets = {'all': list(X.columns), 'suspect_only': [ch, 'Provider Type'],
                'charge_removed': [c for c in X.columns if c not in (ch, 'PCR', 'Charge_Allowed_Ratio')],
                'all_suspects_removed': [c for c in X.columns if c not in (ch, 'Provider Type', 'PCR', 'Charge_Allowed_Ratio')]}
    else:
        X, y = loadB()
        sets = {'all': list(X.columns), 'suspect_only': ['ReimbSum'], 'all_suspects_removed': [c for c in X.columns if c != 'ReimbSum']}
    pw = (1 - y.mean()) / y.mean()
    for s in range(5):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=s)
        for name, cols in sets.items():
            for mn, m in models(s, pw).items():
                _, r = evaluate(m, Xtr[cols], ytr, Xte[cols], yte, s)
                r.update(data=which, set=name, model=mn, seed=s); out.append(r)
                print(which, name, mn, s, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    json.dump(out, open(f'results/robust_perf_{which}.json', 'w'), indent=1)
elif mode == 'shapall':
    X, y, ch = loadA(); cols = [c for c in X.columns if c not in (ch, 'Provider Type', 'PCR', 'Charge_Allowed_Ratio')]
    pw = (1 - y.mean()) / y.mean()
    for s in range(5):
        Xtr, Xte, ytr, yte = train_test_split(X[cols], y, test_size=0.2, stratify=y, random_state=s)
        Xs = Xte.sample(2000, random_state=s)
        for mn, m in models(s, pw).items():
            mf = clone(m).fit(Xtr, ytr)
            sv = shap.TreeExplainer(mf).shap_values(Xs if mn != 'RF' else Xs.iloc[:500])
            if isinstance(sv, list): sv = sv[1]
            sv = np.asarray(sv); sv = sv[:, :, 1] if sv.ndim == 3 else sv
            imp = np.abs(sv).mean(0); imp = imp / imp.sum(); o = np.argsort(-imp)
            r = dict(model=mn, seed=s, top1=cols[o[0]], top1_share=float(imp[o[0]]), top2=cols[o[1]], top2_share=float(imp[o[1]]))
            out.append(r); print(r, flush=True)
    json.dump(out, open('results/robust_shapall.json', 'w'), indent=1)
elif mode == 'group':
    raw = pd.read_csv(os.path.join(prep.DATA, 'HealthcareProviders.csv'), usecols=['National Provider Identifier'])
    g = raw['National Provider Identifier'].values
    X, y, ch = loadA(); assert len(X) == len(g)
    sets = {'all': list(X.columns), 'suspect_only': [ch, 'Provider Type'],
            'charge_removed': [c for c in X.columns if c not in (ch, 'PCR', 'Charge_Allowed_Ratio')]}
    pw = (1 - y.mean()) / y.mean()
    for s, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, y, g)):
        assert not set(g[tr]) & set(g[te])
        for name, cols in sets.items():
            m = models(s, pw)['LGBM']
            _, r = evaluate(m, X.iloc[tr][cols], y.iloc[tr], X.iloc[te][cols], y.iloc[te], s)
            r.update(set=name, fold=s); out.append(r); print(name, s, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    json.dump(out, open('results/robust_group.json', 'w'), indent=1)
