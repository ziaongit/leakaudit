# leakaudit

**Is your model learning the task, or the rule that made the labels?**

`leakaudit` is a small Python package that checks whether a tabular classifier's labels leak from
its input features. It combines SHAP and LIME explanations with simple retraining tests. It
accompanies the paper *"Is Your Fraud Model Learning Fraud or the Labelling Rule? An
Explanation-Based Leakage Audit with SHAP and LIME"* (Ullah & Khan, under review).

---

## 1. Requirements

- Python **3.9 or newer**. Check with `python --version` (on Windows you can also use `py --version`).
- `pip` (it comes with Python).
- `git` is needed only for option B below.

## 2. Installation

It is best to install into a fresh virtual environment so the package does not clash with other projects.

**Create and activate a virtual environment**

Windows (Command Prompt or PowerShell):
```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Option A: install directly from GitHub (recommended)

Full install, with LightGBM, XGBoost and LIME:
```bash
pip install "leakaudit[boost,lime] @ git+https://github.com/ziaongit/leakaudit"
```

Minimal install (Random Forest + SHAP only, no LIME step):
```bash
pip install git+https://github.com/ziaongit/leakaudit
```

### Option B: download the code and install it locally

```bash
git clone https://github.com/ziaongit/leakaudit.git
cd leakaudit
pip install ".[boost,lime]"
```

No git? Click the green **Code** button on GitHub, choose **Download ZIP**, unzip it, open a
terminal in the unzipped folder, and run `pip install ".[boost,lime]"`.

### Check that it worked

```bash
python -c "import leakaudit; print(leakaudit.__version__)"
```

It should print `0.1.0`.

## 3. Quick start

Save this as `quick.py` and run `python quick.py`:

```python
import numpy as np, pandas as pd
from leakaudit import audit

# Synthetic data: the label is built from 'charge' within each 'group' (a leaky z-score rule)
rng = np.random.default_rng(0); n = 5000
X = pd.DataFrame({"charge": rng.lognormal(5, 1, n),
                  "group": rng.integers(0, 5, n),
                  "visits": rng.poisson(20, n)})
z = X.groupby("group").charge.transform(lambda s: (s - s.mean()) / s.std())
y = (z > 2).astype(int)

report = audit(X, y, suspects=["charge", "group"], seeds=(0, 1), lime_records=4)
print(report["verdict"])
```

The last line of the output should start with `LIKELY LEAKAGE`.

A longer demo, with one leaky and one honest label, is in the repository:
```bash
python examples/synthetic_demo.py
```

## 4. Using it on your own data

```python
import pandas as pd
from leakaudit import audit

df = pd.read_csv("my_data.csv")
y = df.pop("label")                  # the 0/1 target column
X = df.select_dtypes("number")       # numeric features (encode categorical columns first)

report = audit(X, y,
               suspects=["feature_used_to_make_label"],   # optional
               derived=["feature_computed_from_it"],       # optional
               lime_records=30)                            # 0 = skip the LIME step
```

- `suspects`: features you think were used to make the label. If you leave it out, the top-ranked
  SHAP feature is used.
- `derived`: features calculated from the suspects. They are removed together with the suspects in Step 3.
- `seeds`: the train/test splits to repeat over. The default `(0, 1, 2, 3, 4)` gives 5 splits.
- `models`: your own `{name: classifier}` dictionary. The default is LightGBM, XGBoost and Random Forest.

`report` is a dictionary. Its main keys are `verdict`, `summary` (a table per model),
`mean_share` (SHAP shares), `stability_across_models` and `lime`.

## 5. The five steps

| Step | What it does | Warning sign |
|---|---|---|
| 1. Attribution concentration | Share of mean \|SHAP\| per feature | One feature dominates in every run |
| 2. Suspect-only model | Retrain on the suspect features only | Matches the full model (**decisive test**) |
| 3. Removal and proxy check | Drop suspects, retrain, re-run SHAP | Large F1 drop; a related feature takes over |
| 4. Global stability | Spearman, Kendall, top-5 Jaccard across splits and models | Very high stability around the suspects |
| 5. Local check (LIME) | Repeat LIME on the same records | Same top-1 everywhere; the rest unstable |

### Reading the verdict

- **LIKELY LEAKAGE**: the suspects alone match the full model, and removing them causes a large drop. Check how the label was made.
- **POSSIBLE LEAKAGE**: the suspects alone match the full model, but other features carry the same information. Look at correlated features.
- **NO SIGN OF LEAKAGE**: the suspects alone do worse than the full model.
- **INCONCLUSIVE**: the full model is close to chance (ROC-AUC < 0.6), so there is little to audit.

## 6. Reproducing the paper

The `paper/` folder contains the scripts that produce every table and figure in the paper. See
`paper/README.md`. The two datasets are public on Kaggle:
[Healthcare Providers Data](https://www.kaggle.com/datasets/tamilsel/healthcare-providers-data) and
[Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis).

## 7. Notes and troubleshooting

- F1 is reported at the best threshold on the test split. This is optimistic and meant only for comparing feature sets.
- A dominant feature alone is not proof of leakage. Steps 2 and 3 decide.
- Runs on large data can take several minutes, mainly because of SHAP. Use fewer `seeds` for a quick check.
- `ModuleNotFoundError: lime`: install the extra with `pip install lime`, or set `lime_records=0`.
- `'git' is not recognized`: use the Download ZIP route in Option B, or install Git from https://git-scm.com.

## License

MIT
