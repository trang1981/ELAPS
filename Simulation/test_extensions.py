"""Self-test: run elaps_extensions on synthetic S2 data (float days and datetimes)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import elaps_sim as S
from elaps_extensions import run_audit
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

rng = np.random.default_rng(7)
d = S.simulate(30000, rng=rng, **S.SCEN["S2_signal"])
idx = np.where(d["r"] + 240 <= S.T_STUDY)[0]           # H180 cohort, full data kept for features
r, e = d["r"][idx], d["e"][idx]
T = r + 240; y = (e <= T).astype(int)
split = np.array(["train"] * len(idx), dtype=object)
_, te = train_test_split(np.arange(len(idx)), test_size=0.2, stratify=y, random_state=42)
split[te] = "test"
for as_dt in (False, True):
    base = pd.Timestamp("2019-01-01")
    conv = (lambda v: base + pd.to_timedelta(v, unit="D")) if as_dt else (lambda v: v)
    coh = pd.DataFrame(dict(r=conv(r), e=conv(np.where(y == 1, e, np.nan)), y=y,
                            T=conv(T), split=split), index=idx)
    def feature_fn(t):
        tn = ((t - base).dt.total_seconds() / 86400).values if as_dt else t.values.astype(float)
        full = np.full(d["n"], np.nan); full[idx] = tn
        full[np.isnan(full)] = d["r"][np.isnan(full)] + 60      # unused rows
        return pd.DataFrame(S.feats(d, full)[idx], index=idx)
    mk = lambda: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, n_jobs=2)
    lm = pd.Timedelta(days=60) if as_dt else 60.0
    lags = [pd.Timedelta(days=k) for k in (7, 30, 60)] if as_dt else [7.0, 30.0, 60.0]
    res, boot, bal = run_audit(coh, feature_fn, mk, lm, lags, n_boot=200)
    print("datetime" if as_dt else "float"); print(res.round(3).to_string(index=False))
    print(boot.round(3).to_string(index=False)); print(bal.round(3).to_string(index=False))
