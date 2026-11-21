"""
ELAPS extensions for the real-data revision (VIB-labeled and Santander).

Adds, on top of the existing A / B-sym / B notebooks:
  1. Bsym_rs   : entry-time (risk-set) matched pseudo-cutoffs for negatives
  2. lag curve : positive cutoff moved to e - k (k in days, or months for Santander)
  3. B->A      : model trained on event-anchored features, scored on day-60 features
                 (likewise Bsym_rs->A): the "deployment cost"
  4. paired customer-level bootstrap on shared holdout resamples
  5. duration-balance report (truncation rate, KS distance) for every pseudo-cutoff rule

The module does not touch feature engineering. You pass your own feature builder
(the function your notebook already uses to compute x(c, t) with strict-before cutoffs).

Minimal use inside 13_VIB60ABC.ipynb (or 09_Santander_ELAPS.ipynb):

    from elaps_extensions import run_audit
    res, boot, bal = run_audit(
        cohort,                       # DataFrame indexed by customer id
        feature_fn=build_features,    # f(cutoffs: pd.Series) -> pd.DataFrame (same index)
        model_fn=make_xgb_pipeline,   # () -> unfitted sklearn-compatible estimator/pipeline
        landmark=pd.Timedelta(days=60),
        lags=[pd.Timedelta(days=k) for k in (7, 30, 60)],   # Santander: pd.DateOffset(months=k)
        n_boot=1000, seed=42)

`cohort` columns:
    r      relationship start (datetime or float days)
    e      first recorded event (NaT/NaN for negatives)
    y      0/1 outcome for this cohort (primary or H180)
    T      admissible endpoint for this cohort (admin end, or r + 240 days for H180)
    split  'train' / 'test'  (reuse the canonical seed-42 split; do NOT re-split)
Row order must be the canonical customer order (pseudo-durations depend on it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


# ----------------------------------------------------------------- helpers
def _num(s: pd.Series) -> np.ndarray:
    """Numeric view of a time-like series (ns for datetimes, raw for floats)."""
    if np.issubdtype(s.dtype, np.datetime64):
        return s.values.astype("datetime64[ns]").astype("int64").astype(float)
    if np.issubdtype(s.dtype, np.timedelta64):
        return s.values.astype("timedelta64[ns]").astype("int64").astype(float)
    return s.values.astype(float)


def _shift(t: pd.Series, k):
    """t - k for Timedelta / DateOffset / float lags."""
    return t - k


# ----------------------------------------------------------------- cutoffs
def cutoff_A(c, landmark):
    return c["r"] + landmark


def cutoff_B(c):
    return c["e"].where(c["y"] == 1, c["T"])


def cutoff_Bsym(c, rng, delays=None):
    """Original B-sym: negatives draw a pseudo-duration from training-positive delays."""
    pos_tr = (c["y"] == 1) & (c["split"] == "train")
    if delays is None:
        delays = (c.loc[pos_tr, "e"] - c.loc[pos_tr, "r"]).values
    neg = c["y"] == 0
    P = rng.choice(delays, neg.sum(), replace=True)
    t = c["e"].copy()
    raw = c.loc[neg, "r"] + P
    t.loc[neg] = raw.where(raw <= c.loc[neg, "T"], c.loc[neg, "T"])
    trunc = pd.Series(False, index=c.index)
    trunc.loc[neg] = (raw > c.loc[neg, "T"]).values
    return t, trunc


def cutoff_Bsym_rs(c, rng, n_strata=10, pos_cut=None):
    """Entry-time matched B-sym: each negative borrows the delay of a random
    training positive from the same relationship-start stratum (risk-set logic).
    pos_cut: optional positive cutoff (e.g. e - k for the lag curve)."""
    pos_cut = c["e"] if pos_cut is None else pos_cut
    pos_tr = (c["y"] == 1) & (c["split"] == "train")
    r_num = _num(c["r"])
    edges = np.quantile(r_num[pos_tr.values], np.linspace(0, 1, n_strata + 1)[1:-1])
    strata = np.digitize(r_num, edges)
    delays = (pos_cut - c["r"])
    neg = (c["y"] == 0).values
    t = pos_cut.copy()
    trunc = pd.Series(False, index=c.index)
    for b in range(n_strata):
        pool = delays[pos_tr.values & (strata == b)].values
        m = neg & (strata == b)
        if m.sum() == 0:
            continue
        if len(pool) == 0:                     # empty stratum: fall back to all positives
            pool = delays[pos_tr.values].values
        P = rng.choice(pool, m.sum(), replace=True)
        raw = c.loc[m, "r"] + P
        T = c.loc[m, "T"]
        t.loc[m] = raw.where(raw <= T, T)
        trunc.loc[m] = (raw > T).values
    return t, trunc


def cutoff_lag(c, k, rng, landmark, n_strata=10):
    """Positives at e - k (never before r + 1 unit); negatives entry-time matched
    to the shifted positive delays, so durations stay comparable."""
    pos_cut = _shift(c["e"], k)
    one = pd.Timedelta(days=1) if np.issubdtype(c["r"].dtype, np.datetime64) else 1.0
    floor = c["r"] + one
    pos_cut = pos_cut.where(pos_cut > floor, floor)
    pos_cut = pos_cut.where(c["y"] == 1, c["T"])
    return cutoff_Bsym_rs(c, rng, n_strata, pos_cut=pos_cut)


# ----------------------------------------------------------------- diagnostics
def balance_report(c, t, trunc, name):
    """Positive event delays vs realised negative durations (post-truncation)."""
    pos, neg = c["y"] == 1, c["y"] == 0
    scale = 86400e9 if np.issubdtype(c["r"].dtype, np.datetime64) else 1.0  # -> days
    d_pos = _num(t[pos] - c.loc[pos, "r"]) / scale
    d_neg = _num(t[neg] - c.loc[neg, "r"]) / scale
    return dict(branch=name, trunc_rate=float(trunc[neg].mean()),
                median_pos=np.median(d_pos), median_neg=np.median(d_neg),
                ks=ks_2samp(d_pos, d_neg).statistic)


def paired_bootstrap(y, preds: dict, pairs, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    need = sorted({x for p in pairs for x in p})
    draws = {f"{a} - {b}": [] for a, b in pairs}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        auc = {k: roc_auc_score(y[i], preds[k][i]) for k in need}
        for a, b in pairs:
            draws[f"{a} - {b}"].append(auc[a] - auc[b])
    full = {k: roc_auc_score(y, preds[k]) for k in need}
    rows = []
    for (a, b) in pairs:
        v = np.array(draws[f"{a} - {b}"])
        rows.append(dict(contrast=f"{a} - {b}", estimate=full[a] - full[b],
                         lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- main audit
def run_audit(cohort, feature_fn, model_fn, landmark, lags, n_boot=1000,
              seed=42, n_strata=10):
    c = cohort.copy()
    rng = np.random.default_rng(seed)
    tr = (c["split"] == "train").values
    te = ~tr
    y = c["y"].values

    cuts, truncs = {}, {}
    cuts["A"] = cutoff_A(c, landmark)
    cuts["B"] = cutoff_B(c)
    cuts["Bsym"], truncs["Bsym"] = cutoff_Bsym(c, rng)
    cuts["Bsym_rs"], truncs["Bsym_rs"] = cutoff_Bsym_rs(c, rng, n_strata)
    for k in lags:
        name = f"lag{getattr(k, 'days', None) or getattr(k, 'months', None) or k}"
        cuts[name], truncs[name] = cutoff_lag(c, k, rng, landmark, n_strata)

    X = {b: feature_fn(t).loc[c.index] for b, t in cuts.items()}
    preds, rows = {}, []
    for b, Xb in X.items():
        m = model_fn().fit(Xb.iloc[tr], y[tr])
        preds[b] = m.predict_proba(Xb.iloc[te])[:, 1]
        if b in ("B", "Bsym_rs"):
            preds[f"{b}->A"] = m.predict_proba(X["A"].iloc[te])[:, 1]
    for k, p in preds.items():
        rows.append(dict(branch=k, auc=roc_auc_score(y[te], p),
                         brier=np.mean((p - y[te]) ** 2)))
    res = pd.DataFrame(rows)
    prev = y[tr].mean()
    bs_ref = np.mean((prev - y[te]) ** 2)
    res["bss"] = 1 - res["brier"] / bs_ref

    lag_names = [b for b in cuts if b.startswith("lag")]
    pairs = [("B", "A"), ("Bsym_rs", "A"), ("B", "Bsym_rs"), ("Bsym", "A"),
             ("B->A", "A"), ("Bsym_rs->A", "A")] + [(l, "A") for l in lag_names]
    boot = paired_bootstrap(y[te], preds, pairs, n_boot, seed)
    bal = pd.DataFrame([balance_report(c, cuts[b], truncs[b], b) for b in truncs])
    return res, boot, bal
