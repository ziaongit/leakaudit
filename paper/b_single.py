import numpy as np, lightgbm as lgb, prep_ext
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
B=prep_ext.load_ext(verbose=False); y=B.pop('Fraud')
def best(y,p): return max(f1_score(y,p>=t) for t in np.linspace(.05,.95,19))
for cols in [['ReimbSum'],['ReimbSum','NClaims'],[c for c in B.columns if c!='ReimbSum'],list(B.columns)]:
    f,a=[],[]
    for s in range(5):
        Xtr,Xte,ytr,yte=train_test_split(B[cols],y,test_size=.2,stratify=y,random_state=s)
        m=lgb.LGBMClassifier(n_estimators=300,learning_rate=.05,random_state=s,verbose=-1,scale_pos_weight=(1-y.mean())/y.mean()).fit(Xtr,ytr)
        p=m.predict_proba(Xte)[:,1]; f.append(best(yte,p)); a.append(roc_auc_score(yte,p))
    print(cols if len(cols)<3 else f'{len(cols)} feats', f'F1 {np.mean(f):.3f}±{np.std(f,ddof=1):.3f} AUC {np.mean(a):.3f}')
