# leakaudit

**Is your model learning the task, or the rule that made the labels?**

`leakaudit` is a small Python package that checks whether a tabular classifier's labels leak from
its input features. It uses SHAP and LIME explanations together with simple retraining tests. It
accompanies the paper *"Is Your Fraud Model Learning Fraud or the Labelling Rule? An
Explanation-Based Leakage Audit with SHAP and LIME"* (Ullah & Khan, under review).

## The five steps

| Step | What it does | Warning sign |
|---|---|---|
| 1. Attribution concentration | Share of mean \|SHAP\| per feature | One feature dominates in every run |
| 2. Suspect-only model | Retrain on the suspect features only | Matches the full model (**decisive test**) |
| 3. Removal and proxy check | Drop suspects, retrain, re-run SHAP | Large F1 drop; a related feature takes over |
| 4. Global stability | Spearman, Kendall, top-5 Jaccard across splits and models | Very high stability around the suspects |
| 5. Local check (LIME) | Repeat LIME on the same records | Same top-1 everywhere; the rest unstable |

## Install

```bash
pip install .            # core (scikit-learn Random Forest + SHAP)
pip install ".[boost]"   # adds LightGBM and XGBoost
pip install ".[lime]"    # adds LIME for Step 5
```

## Use

```python
from leakaudit import audit
report = audit(X, y, suspects=["Average Submitted Charge Amount", "Provider Type"],
               derived=["PCR", "Charge_Allowed_Ratio"], lime_records=30)
print(report["verdict"])
```

If `suspects` is not given, the top-ranked SHAP feature from Step 1 is used. Run
`python examples/synthetic_demo.py` for a demo on synthetic data.

## Notes

- F1 is reported at the best threshold on the test split. This is optimistic and meant only for
  comparisons between feature sets.
- A dominant feature alone is not proof of leakage; Steps 2 and 3 decide.
- The `paper/` folder holds the scripts that reproduce every table and figure in the paper.

## License

MIT
