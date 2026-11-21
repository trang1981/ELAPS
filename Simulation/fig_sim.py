import sys, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = pd.read_csv(sys.argv[1])
br = ["A", "Bsym_rs", "B", "lag7", "lag30", "lag60", "B->A"]
lab = ["A", "B-sym-rs", "B", "lag 7", "lag 30", "lag 60", "B→A"]
sc = [("S0_null", "Null: no signal"), ("S1_ramp", "Ramp: 30-day pre-event ramp"), ("S2_signal", "Signal: signal + ramp")]
co = [("H180", "Equal 180-day follow-up"), ("primary", "Common administrative endpoint")]
g = d.groupby(["scenario", "cohort"])[br]; m, s = g.mean(), g.std()
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
for i, (c, ct) in enumerate(co):
    for j, (k, kt) in enumerate(sc):
        a = ax[i, j]; mm, ss = m.loc[(k, c)].values, s.loc[(k, c)].values
        a.axhline(m.loc[(k, c), "A"], color="#9a9a9a", lw=1, ls="--", zorder=0)
        a.errorbar(range(len(br)), mm, yerr=ss, fmt="o", ms=4.5, color="#1f4e79",
                   ecolor="#1f4e79", elinewidth=1, capsize=0)
        a.set_ylim(0.4, 1.02); a.grid(axis="y", color="#e6e6e6", lw=0.6); a.set_axisbelow(True)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        if i == 0: a.set_title(kt, fontsize=9)
        if j == 0: a.set_ylabel(f"{ct}\nholdout AUC", fontsize=8.5)
        a.set_xticks(range(len(br))); a.set_xticklabels(lab, rotation=45, ha="right")
fig.text(0.5, 0.005, "Dashed line: Branch A (deployment-aligned). Points: mean of 5 replicates; bars: ±1 SD.\n"
         "True deployable AUC is 0.50 in the null scenario; lag cutoffs are floored at day 60.", ha="center", fontsize=7.5, color="#444")
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(sys.argv[2], dpi=300)
