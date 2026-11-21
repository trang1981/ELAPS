"""
ELAPS synthetic positive control (licence-free, ground truth known).

Scenarios
  S0_null    : adoption hazard independent of every feature, no pre-adoption ramp.
               Any AUC > 0.5 is pure evaluation artefact (true AUC = 0.5).
  S1_ramp    : no stable propensity signal, but digital activity rises in the
               R days before adoption (event-proximate signal = anchoring leak).
  S2_signal  : true propensity (hazard depends on latent digital intensity) + ramp.

Cohorts
  primary : common administrative endpoint (day 365) -> heterogeneous follow-up.
  H180    : 180-day post-landmark follow-up for everyone (r + 240 <= 365).

Branches (feature-window rules; strict-before cutoffs)
  A          : t = r + 60
  B          : pos t = e ; neg t = T
  Bsym       : pos t = e ; neg t = min(r + P, T), P ~ training-positive delays
  Bsym_cond  : as Bsym but P drawn only from delays admissible for that negative
               (P <= T - r), so no truncation
  lag_k      : pos t = e - k ; neg conditional pseudo-cutoff from (e - k - r)
  B->A       : model trained on B features, holdout scored with day-60 features
  Bsym_rs    : negatives borrow delays of training positives from the same
               entry-time stratum (risk-set / incidence-density matching)
  Bsymrs->A  : model trained on Bsym_rs features, scored with day-60 features
  lag_k uses the risk-set matched construction for negatives

Usage: python3 elaps_sim.py --reps 5 --n 60000 --model xgb --out results_xgb.csv
Model: XGBoost with the canonical ELAPS settings (300 trees, depth 6, lr 0.05,
subsample 0.8, colsample 0.8); paired 1,000-resample bootstrap on replicate 0.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

T_STUDY = 365.0
L = 60.0
KEY = 1000.0
SCEN = {
    "S0_null":   dict(beta=0.0, rho=0.0, h0=0.00033),
    "S1_ramp":   dict(beta=0.0, rho=0.1, h0=0.00033),
    "S2_signal": dict(beta=0.8, rho=0.1, h0=0.00025),
}
RAMP_DAYS = 30.0
LAGS = (7, 30, 60)


def simulate(n, beta, rho, h0, rng):
    r = rng.uniform(0, 300, n)
    lam_t = rng.gamma(2.0, 0.25, n)              # transactions / day
    lam_d = rng.gamma(2.0, 0.10, n)              # digital sessions / day
    z = (np.log(lam_d) - np.log(lam_d).mean()) / np.log(lam_d).std()
    h = h0 * np.exp(beta * z)
    e = r + rng.exponential(1.0 / h)
    keep = e > r + L                             # eligibility: no event by landmark
    r, lam_t, lam_d, e = r[keep], lam_t[keep], lam_d[keep], e[keep]
    n = len(r)
    cid = np.arange(n)

    def stream(rate, extra=None):
        span = T_STUDY - r
        k = rng.poisson(rate * span)
        c = np.repeat(cid, k)
        t = r[c] + rng.uniform(0, 1, k.sum()) * span[c]
        if extra is not None:
            c2, t2 = extra
            c, t = np.concatenate([c, c2]), np.concatenate([t, t2])
        key = np.sort(c * KEY + t)
        return key

    ramp = None
    if rho > 0:
        has = e < T_STUDY
        lo = np.maximum(r, e - RAMP_DAYS)
        span = np.where(has, e - lo, 0.0)
        k = rng.poisson(rho * span)
        c2 = np.repeat(cid, k)
        t2 = lo[c2] + rng.uniform(0, 1, k.sum()) * span[c2]
        ramp = (c2, t2)
    return dict(r=r, e=e, txn=stream(lam_t), dig=stream(lam_d, ramp), n=n)


def window_feats(key, r, t):
    cid = np.arange(len(r))
    base = cid * KEY
    i_r = np.searchsorted(key, base + r, "left")
    i_t = np.searchsorted(key, base + t, "left")
    i_30 = np.searchsorted(key, base + np.maximum(r, t - 30), "left")
    cnt = (i_t - i_r).astype(float)
    last = i_t - 1
    has = last >= i_r
    since = np.where(has, t - (key[np.clip(last, 0, None)] - base), t - r + 1)
    ln = np.maximum(t - r, 1e-9)
    return np.column_stack([cnt, cnt / ln, i_t - i_30, since])


def feats(d, t):
    return np.hstack([window_feats(d["txn"], d["r"], t),
                      window_feats(d["dig"], d["r"], t)])


def pseudo(delays_pool, r, T, rng, conditional):
    if not conditional:
        P = rng.choice(delays_pool, len(r), replace=True)
        return np.minimum(r + P, T), (r + P > T)
    pool = np.sort(delays_pool)
    lim = T - r
    m = np.searchsorted(pool, lim, "right")          # admissible count
    u = rng.integers(0, np.maximum(m, 1))
    P = np.where(m > 0, pool[np.minimum(u, len(pool) - 1)], lim)
    return r + P, np.zeros(len(r), bool)


def pseudo_rs(delays, r_pos, r, T, rng, nbins=10):
    """Entry-time (risk-set) matched pseudo-cutoff: each negative borrows the
    delay of a random training positive from the same entry-time stratum."""
    edges = np.quantile(r_pos, np.linspace(0, 1, nbins + 1)[1:-1])
    sp, sn = np.digitize(r_pos, edges), np.digitize(r, edges)
    P = np.empty(len(r))
    for b in range(nbins):
        pool = delays[sp == b]
        m = sn == b
        P[m] = rng.choice(pool, m.sum(), replace=True)
    return np.minimum(r + P, T), (r + P > T)


MODEL = "xgb"


def make_model(seed):
    if MODEL == "xgb":
        from xgboost import XGBClassifier   # canonical ELAPS settings
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", n_jobs=2, random_state=seed)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                          random_state=seed)


def fit_pred(Xtr, ytr, Xte_list, seed):
    m = make_model(seed).fit(Xtr, ytr)
    return [m.predict_proba(X)[:, 1] for X in Xte_list]


def paired_boot(yte, preds, pairs, B=1000, seed=42):
    """Customer-level paired bootstrap: same resample indices for all branches."""
    rng = np.random.default_rng(seed)
    n = len(yte)
    out = {f"{a}-{b}": [] for a, b in pairs}
    for _ in range(B):
        i = rng.integers(0, n, n)
        if yte[i].min() == yte[i].max():
            continue
        need = {x for pr in pairs for x in pr}
        auc = {k: roc_auc_score(yte[i], preds[k][i]) for k in need}
        for a, b in pairs:
            out[f"{a}-{b}"].append(auc[a] - auc[b])
    return {k: (np.mean(v), *np.percentile(v, [2.5, 97.5])) for k, v in out.items()}


BOOT = False


def run_cohort(d, cohort, rng, seed):
    r, e = d["r"], d["e"]
    if cohort == "primary":
        idx = np.arange(d["n"])
        T = np.full(d["n"], T_STUDY)
        y = (e < T_STUDY).astype(int)
    else:  # H180
        idx = np.where(r + 240 <= T_STUDY)[0]
        T = r[idx] + 240
        y = (e[idx] <= T).astype(int)
    sub = {k: (v[idx] if k in ("r", "e") else v) for k, v in d.items()}
    # remap streams to subset by rebuilding keys (cheap: filter + recode)
    for s in ("txn", "dig"):
        c = np.floor(d[s] / KEY).astype(int)
        mask = np.isin(c, idx)
        newc = np.searchsorted(idx, c[mask])
        sub[s] = np.sort(newc * KEY + (d[s][mask] - c[mask] * KEY))
    r, e = sub["r"], sub["e"]
    n = len(r)
    tr, te = train_test_split(np.arange(n), test_size=0.2, stratify=y,
                              random_state=seed)
    pos_tr = tr[y[tr] == 1]
    delays = e[pos_tr] - r[pos_tr]

    cut = {"A": r + L}
    cut["B"] = np.where(y == 1, e, T)
    t_s, trunc = pseudo(delays, r, T, rng, conditional=False)
    cut["Bsym"] = np.where(y == 1, e, t_s)
    t_c, _ = pseudo(delays, r, T, rng, conditional=True)
    cut["Bsym_cond"] = np.where(y == 1, e, t_c)
    t_rs, trunc_rs = pseudo_rs(delays, r[pos_tr], r, T, rng)
    cut["Bsym_rs"] = np.where(y == 1, e, t_rs)
    for k in LAGS:
        tp = np.maximum(e - k, r + 1)
        dl = tp[pos_tr] - r[pos_tr]
        t_k, _ = pseudo_rs(dl, r[pos_tr], r, T, rng)
        cut[f"lag{k}"] = np.where(y == 1, tp, t_k)

    X = {b: feats(sub, c) for b, c in cut.items()}
    out, preds = {}, {}
    for b in cut:
        p_own, p_A = fit_pred(X[b][tr], y[tr], [X[b][te], X["A"][te]], seed)
        preds[b] = p_own
        if b == "B":
            preds["B->A"] = p_A
        if b == "Bsym_rs":
            preds["Bsymrs->A"] = p_A
    bs_ref = np.mean((y[tr].mean() - y[te]) ** 2)
    for k, v in preds.items():
        out[k] = roc_auc_score(y[te], v)
        out[f"bss_{k}"] = 1 - np.mean((v - y[te]) ** 2) / bs_ref
    if BOOT:
        pairs = [("B", "A"), ("Bsym_rs", "A"), ("B", "Bsym_rs"),
                 ("lag30", "A"), ("B->A", "A")]
        for k, (m_, lo, hi) in paired_boot(y[te], preds, pairs).items():
            out[f"boot[{k}]"] = f"{m_:.3f} ({lo:.3f} to {hi:.3f})"
    neg = y == 0
    out.update(n=n, prev=y.mean(), trunc_rate=trunc[neg].mean(),
               trunc_rs=trunc_rs[neg].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--model", default="xgb", choices=["xgb", "hgb"])
    a = ap.parse_args()
    global MODEL, BOOT
    MODEL = a.model
    rows = []
    for sc, par in SCEN.items():
        for rep in range(a.reps):
            rng = np.random.default_rng(1000 + rep)
            d = simulate(a.n, rng=rng, **par)
            BOOT = rep == 0          # paired bootstrap on the first replicate
            for cohort in ("primary", "H180"):
                o = run_cohort(d, cohort, rng, seed=42 + rep)
                o.update(scenario=sc, cohort=cohort, rep=rep)
                rows.append(o)
                print(sc, cohort, rep, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    cols = ["n", "prev", "trunc_rate", "trunc_rs", "A", "Bsym", "Bsym_cond",
            "Bsym_rs", "B", "lag7", "lag30", "lag60", "B->A", "Bsymrs->A",
            "bss_A", "bss_Bsym_rs", "bss_B", "bss_B->A", "bss_Bsymrs->A"]
    g = df.groupby(["scenario", "cohort"])[cols]
    summ = g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)
    summ.to_csv(a.out.replace(".csv", "_summary.csv"))
    print(g.mean().round(3).to_string())


if __name__ == "__main__":
    main()
