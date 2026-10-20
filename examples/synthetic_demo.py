"""Demo: a label built from two features is caught; a label from hidden signal is not."""
import numpy as np, pandas as pd
from leakaudit import audit

rng = np.random.default_rng(0)
n = 20000
X = pd.DataFrame({"charge": rng.lognormal(5, 1, n), "group": rng.integers(0, 10, n),
                  "visits": rng.poisson(20, n), "age": rng.normal(70, 8, n), "noise": rng.normal(size=n)})
# Leaky label: charge far above its group's mean (a z-score rule, like a common Medicare practice)
z = X.groupby("group").charge.transform(lambda s: (s - s.mean()) / s.std())
y_leaky = (z > 2).astype(int)
audit(X, y_leaky, suspects=["charge", "group"], seeds=(0, 1))

# Non-leaky label: several inputs each carry part of the signal, plus unobserved noise
hidden = (X.visits - 20) / 4.5 + (X.age - 70) / 8 + 0.5 * (np.log(X.charge) - 5) + rng.normal(size=n)
y_ok = (hidden > np.quantile(hidden, 0.95)).astype(int)
audit(X, y_ok, seeds=(0, 1))
