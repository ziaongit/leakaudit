import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
def load(path='/mnt/user-data/uploads/PhD/Datasets/HealthcareProviders.csv', corr_filter=True, verbose=True):
    df=pd.read_csv(path)
    drop_cols=[c for c in df.columns if any(k in c.lower() for k in ['npi','name','address','city','zip','street'])]
    df.drop(columns=drop_cols,inplace=True,errors='ignore')
    for col in df.columns:
        if df[col].dtype==object or pd.api.types.is_string_dtype(df[col]):
            conv=pd.to_numeric(df[col].str.replace(',',''),errors='coerce')
            if conv.notna().mean()>0.5: df[col]=conv
    cat_cols=df.select_dtypes(include=['object','string']).columns.tolist()
    for col in cat_cols:
        df[col]=LabelEncoder().fit_transform(df[col].fillna('Unknown').astype(str))
    charge=[c for c in df.columns if 'submitted' in c.lower() and 'charge' in c.lower()][0]
    pay=[c for c in df.columns if 'payment' in c.lower() and 'average' in c.lower()][0]
    allowed=[c for c in df.columns if 'allowed' in c.lower() and 'average' in c.lower()][0]
    serv=[c for c in df.columns if 'number of services' in c.lower()][0]
    bene=[c for c in df.columns if 'number of medicare beneficiar' in c.lower()][0]
    for col in [pay,allowed,serv,bene]: df[col]=df[col].fillna(df[col].median())
    df['PCR']=(df[pay]/df[charge].replace(0,np.nan)).clip(0,1)
    df['Charge_Allowed_Ratio']=(df[charge]/df[allowed].replace(0,np.nan)).clip(0,10)
    df['Services_Per_Beneficiary']=(df[serv]/df[bene].replace(0,np.nan)).clip(0,100)
    df['Payment_Per_Beneficiary']=(df[pay]/df[bene].replace(0,np.nan)).clip(0,1e6)
    pt=[c for c in df.columns if c.lower()=='provider type'] or [c for c in df.columns if 'provider' in c.lower() and 'type' in c.lower()]
    ptc=pt[0]
    st=df[df[charge].notna()].groupby(ptc)[charge].agg(['mean','std']).reset_index()
    st.columns=[ptc,'m','s']
    df=df.merge(st,on=ptc,how='left')
    df['Fraud']=((df[charge]-df['m'])/df['s'].replace(0,np.nan)>2).fillna(False).astype(int)
    df.drop(columns=['m','s'],inplace=True)
    df.fillna(df.median(numeric_only=True),inplace=True)
    dropped=[]
    if corr_filter:
        corr=df.drop(columns=['Fraud']).corr(numeric_only=True).abs()
        up=corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
        dropped=[c for c in up.columns if any(up[c]>0.85)]
        df.drop(columns=dropped,inplace=True,errors='ignore')
    if verbose:
        print('provider type col:',ptc,'| charge col:',charge)
        print('fraud:',df.Fraud.sum(), f'{df.Fraud.mean()*100:.2f}%')
        print('dropped by corr:',dropped)
        print('features:',[c for c in df.columns if c!='Fraud'])
    return df, charge
if __name__=='__main__':
    df,ch=load()
