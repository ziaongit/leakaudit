# Reproducing the paper

These scripts reproduce every table and figure in *"Is Your Fraud Model Learning Fraud or the
Labelling Rule? An Explanation-Based Leakage Audit with SHAP and LIME"*.

## 1. Data

Download both public Kaggle datasets into a folder called `data` inside this `paper` folder:

```
paper/data/HealthcareProviders.csv          # https://www.kaggle.com/datasets/tamilsel/healthcare-providers-data
paper/data/ProviderFraud/Train-*.csv         # https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
paper/data/ProviderFraud/Train_Beneficiarydata-*.csv
paper/data/ProviderFraud/Train_Inpatientdata-*.csv
paper/data/ProviderFraud/Train_Outpatientdata-*.csv
```

To keep the data elsewhere, set `LEAKAUDIT_DATA` to that folder.

## 2. Install

```bash
pip install pandas numpy scipy scikit-learn shap lightgbm xgboost lime matplotlib python-docx
```

## 3. Run (from inside this folder; create `results` and `figs` first)

```bash
mkdir results figs
python audit.py A            # Step 1/4 on dataset A, all features      -> Tables 2, 3; Fig. 3
python audit.py Anocharge    # charge features removed (SHAP, stability) -> Tables 2, 3
python audit.py B            # dataset B                                 -> Tables 2, 3; Fig. 3
python audit.py rule5        # suspect-only SHAP shares (text, Section 5.2)
python audit.py lime         # Step 5 (LIME)                             -> Table 4
python robust.py perf A      # F1/AUC/PR-AUC, validation-tuned threshold -> Table 1, Fig. 2
python robust.py perf B      #                                           -> Table 1, Fig. 2
python robust.py shapall     # all suspects removed (SHAP)               -> Table 2
python robust.py group       # provider-grouped splits                   -> Section 5.5
python figs_p2.py            # figures
python build_p2.py           # manuscript (.docx); tables are filled from results/*.json
```

`b_single.py` is an earlier LightGBM-only version of the dataset B check, kept for reference;
`robust.py perf B` supersedes it. The slowest steps (dataset A with Random Forest SHAP) take
tens of minutes on a laptop.
