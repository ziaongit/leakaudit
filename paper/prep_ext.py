"""Second dataset: Kaggle 'Healthcare Provider Fraud Detection Analysis' (rohitrox).
Provider-level labels (PotentialFraud Yes/No) come with the data and are NOT derived from the features.
Claims are aggregated to one row per provider using the training files only."""
import glob, numpy as np, pandas as pd

def find(d, key):
    f = [p for p in glob.glob(d + '/**/*.csv', recursive=True) if key.lower() in p.lower() and 'test' not in p.lower().split('/')[-1]]
    assert f, f'missing {key} in {d}'; return f[0]

def load_ext(d='/mnt/user-data/uploads/PhD/Datasets/ProviderFraud', verbose=True):
    lab = pd.read_csv([p for p in glob.glob(d + '/**/Train*.csv', recursive=True)
                       if not any(k in p for k in ('Beneficiary', 'Inpatient', 'Outpatient'))][0])
    ben = pd.read_csv(find(d, 'Train_Beneficiary')); ip = pd.read_csv(find(d, 'Train_Inpatient')); op = pd.read_csv(find(d, 'Train_Outpatient'))
    ben['Dead'] = ben['DOD'].notna().astype(int)
    ben['Age'] = (pd.to_datetime('2009-12-01') - pd.to_datetime(ben['DOB'])).dt.days / 365.25
    chron = [c for c in ben.columns if c.startswith('ChronicCond_')]
    for c in chron: ben[c] = (ben[c] == 1).astype(int)
    ben['NChronic'] = ben[chron].sum(1); ben['RenalDisease'] = (ben['RenalDiseaseIndicator'] == 'Y').astype(int)
    ip['Inpatient'] = 1; op['Inpatient'] = 0
    cl = pd.concat([ip, op], ignore_index=True).merge(ben[['BeneID', 'Dead', 'Age', 'NChronic', 'RenalDisease', 'Gender']], on='BeneID', how='left')
    cl['ClaimDays'] = (pd.to_datetime(cl['ClaimEndDt']) - pd.to_datetime(cl['ClaimStartDt'])).dt.days + 1
    cl['AdmitDays'] = (pd.to_datetime(cl['DischargeDt']) - pd.to_datetime(cl['AdmissionDt'])).dt.days + 1
    dx = [c for c in cl.columns if c.startswith('ClmDiagnosisCode_')]; px = [c for c in cl.columns if c.startswith('ClmProcedureCode_')]
    cl['NDiag'] = cl[dx].notna().sum(1); cl['NProc'] = cl[px].notna().sum(1)
    cl['HasOperating'] = cl['OperatingPhysician'].notna().astype(int); cl['HasOther'] = cl['OtherPhysician'].notna().astype(int)
    cl['Deductible'] = cl['DeductibleAmtPaid'].fillna(0)
    g = cl.groupby('Provider')
    X = pd.DataFrame({
        'NClaims': g.size(), 'NBenes': g['BeneID'].nunique(), 'InpatientShare': g['Inpatient'].mean(),
        'ReimbMean': g['InscClaimAmtReimbursed'].mean(), 'ReimbSum': g['InscClaimAmtReimbursed'].sum(), 'ReimbMax': g['InscClaimAmtReimbursed'].max(),
        'DeductMean': g['Deductible'].mean(), 'ClaimDaysMean': g['ClaimDays'].mean(), 'AdmitDaysMean': g['AdmitDays'].mean(),
        'NDiagMean': g['NDiag'].mean(), 'NProcMean': g['NProc'].mean(), 'NAttending': g['AttendingPhysician'].nunique(),
        'OperatingShare': g['HasOperating'].mean(), 'OtherPhysShare': g['HasOther'].mean(),
        'BeneAgeMean': g['Age'].mean(), 'BeneDeadShare': g['Dead'].mean(), 'BeneChronicMean': g['NChronic'].mean(),
        'BeneRenalShare': g['RenalDisease'].mean(), 'BeneMaleShare': g['Gender'].apply(lambda s: (s == 1).mean()),
    })
    X['ClaimsPerBene'] = X['NClaims'] / X['NBenes']
    X = X.reset_index().merge(lab, on='Provider', how='inner')
    X['Fraud'] = (X['PotentialFraud'] == 'Yes').astype(int)
    X = X.drop(columns=['Provider', 'PotentialFraud']).fillna(0)
    if verbose: print('providers', len(X), 'fraud', X.Fraud.sum(), f'{X.Fraud.mean()*100:.2f}%', 'features', X.shape[1] - 1)
    return X
if __name__ == '__main__':
    load_ext()
