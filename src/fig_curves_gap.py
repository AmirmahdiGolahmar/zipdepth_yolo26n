# ── Figure 1: (a) val mAP50-95 per epoch (50 ep + continuation), (b) gap to A ──
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

RUNS = Path("/content/drive/MyDrive/lab/runs")           # same folder as in the notebooks
COL = "metrics/mAP50-95(B)"                             # validation mAP50-95 column in results.csv

def curve(name):
    """Per-epoch val mAP50-95 of one run, read from its results.csv."""
    r = pd.read_csv(RUNS / name / "results.csv")
    r.columns = r.columns.str.strip()
    return r[COL].to_numpy()

# name, colour, run folders (50-epoch run + continuation run, if there is one)
MODELS = {
    "A: YOLO26n":             ("#2a78d6", ["A_yolo26n", "A_yolo26n_ft"]),
    "B-A3: 3-layer adapters": ("#1f9e6e", ["B_zipdepth_a3", "B_zipdepth_a3_ft"]),
    "B: 1-layer adapters":    ("#eb6834", ["B_zipdepth", "B_zipdepth_ft"]),
    "B-RN: random neck/head": ("#8a6fd1", ["B_zipdepth_rn"]),
}
# final evaluation of each best checkpoint (model.val() in section 8), used only for the labels
FINAL_50 = {"A: YOLO26n": 0.588, "B-A3: 3-layer adapters": 0.512, "B: 1-layer adapters": 0.472, "B-RN: random neck/head": 0.461}
FINAL_70 = {"A: YOLO26n": 0.602, "B-A3: 3-layer adapters": 0.531, "B: 1-layer adapters": 0.490}
# gap to A after 50 and 70 epochs, from the comparison tables (section 13); computed from unrounded values
GAP = {"B": (0.116, 0.112), "B-A3": (0.076, 0.070)}

# join the 50 epochs and the continuation into one curve per model
curves = {k: np.concatenate([curve(n) for n in runs if (RUNS / n / "results.csv").exists()])
          for k, (_, runs) in MODELS.items()}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.35, 1]})

# (a) mAP50-95 per epoch
for k, v in curves.items():
    c = MODELS[k][0]
    e = np.arange(1, len(v) + 1)
    ax.plot(e, v, "-", color=c, lw=1.6, label=k)
    ax.plot(e[-1], v[-1], "o", color=c, ms=4)                          # end point
    final = FINAL_70.get(k, FINAL_50[k])
    dy = -0.03 if k.startswith("B-RN") else 0                          # B-RN label below the B curve
    ax.text(e[-1] + 1.5, v[-1] + dy, f"{final:.3f}", va="center", fontsize=8, color=c)
ax.axvspan(50.5, 70.5, color="#f4f3f0", zorder=0)                      # shaded continuation phase
ax.text(60.5, 0.12, "+20 epochs\nlr 1e-3 to 1e-4\nno mosaic\npatience 5", ha="center", fontsize=8, color="#52514e")
ax.set(xlim=(0, 76), ylim=(0, 0.66), xlabel="epoch", ylabel="val mAP50-95")
ax.set_title("(a) Validation mAP50-95 during training", loc="left")
ax.grid(alpha=0.4)

# (b) gap to A = A's curve minus the model's curve, epoch by epoch
a = curves["A: YOLO26n"]
for k in ["B: 1-layer adapters", "B-A3: 3-layer adapters"]:
    v = curves[k]
    n = min(len(a), len(v))                                             # same epochs for both
    gap = a[:n] - v[:n]
    short = k.split(":")[0]
    bx.plot(np.arange(10, n + 1), gap[9:], "-", color=MODELS[k][0], lw=1.6, label=f"gap A vs {short}")
    g50, g70 = GAP[short]
    bx.text(60.5, g70 + (0.045 if short == "B" else -0.035), f"{g50:.3f} to {g70:.3f}",
            ha="center", fontsize=8, color=MODELS[k][0])
bx.axvspan(50.5, 70.5, color="#f4f3f0", zorder=0)
bx.set(xlim=(10, 71), ylim=(0, 0.24), xlabel="epoch", ylabel="gap to A (mAP50-95)")
bx.set_title("(b) Gap to A (label: 50 ep to 70 ep)", loc="left")
bx.legend(loc="upper right", frameon=False)
bx.grid(alpha=0.4)

h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False)            # one shared legend below both panels
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(RUNS.parent / "results" / "fig_curves_gap.png", dpi=200)
plt.show()
