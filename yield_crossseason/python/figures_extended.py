"""
Figures for the extended sweep (spectral, all-parameter and season-specific blocks).

  figX1  best achievable R2 by predictor set x validation scheme
  figX2  individual model performance on the spectral and all-parameter sets
  figX3  predictor-set ranking for all three yield targets
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from wheat_paths import (ROOT, F24, F25, OUT, CSVD, XLSX, RESD, FIGD,
                         ML_DIR, REPORT_DOCX, check_inputs)

S1, S2 = "2024-25", "2025-26"

SETS = ["Design_only", "Physio_only", "Ground_summary", "Ground_ts", "Extras24",
        "Physio_extra24", "Agro_predictive", "Agro_full", "Pheno", "Spectral",
        "All_remote", "All_parameters"]
SCHEMES = ["CV_2024-25", "Block_2024-25", "CV_2025-26", "Block_2025-26", "CV_pooled",
           "Transfer_24to25", "Transfer_25to24", "TransferZ_24to25", "TransferZ_25to24"]
LAB = {"CV_2024-25": "CV 24-25", "Block_2024-25": "Blk 24-25", "CV_2025-26": "CV 25-26",
       "Block_2025-26": "Blk 25-26", "CV_pooled": "Pooled", "Transfer_24to25": "Tr 24>25",
       "Transfer_25to24": "Tr 25>24", "TransferZ_24to25": "TrZ 24>25",
       "TransferZ_25to24": "TrZ 25>24"}
MODELS = ["Dummy_mean", "OLS", "Ridge", "Lasso", "ElasticNet", "PLSR", "SVR_RBF",
          "RandomForest", "ExtraTrees", "HistGB", "GPR", "KNN"]

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "figure.facecolor": "white", "savefig.bbox": "tight"})
C_S1, C_S2 = "#2B6CB0", "#C05621"


def heat(ax, mat, fig, title, vmin=-0.6, vmax=0.9):
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(mat.values, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=32, ha="right")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, ("%.2f" % v) if v > -10 else "<<0", ha="center", va="center",
                        fontsize=7.5, color="black" if -0.25 < v < 0.65 else "white")
    ax.grid(False)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, label="$R^2$", shrink=0.9)


def figX1(res):
    d = res[(res.Target == "Grain_Yield_t_ha") & (res.Model != "Dummy_mean")]
    best = d.sort_values("R2_mean", ascending=False) \
            .groupby(["Feature_set", "Scheme"]).first().reset_index()
    mat = best.pivot(index="Feature_set", columns="Scheme", values="R2_mean")
    mat = mat.reindex(index=[s for s in SETS if s in mat.index],
                      columns=[s for s in SCHEMES if s in mat.columns])
    mat.columns = [LAB.get(c, c) for c in mat.columns]
    fig, ax = plt.subplots(figsize=(11, 5.0), constrained_layout=True)
    heat(ax, mat, fig, "Grain yield: best achievable $R^2$ by predictor set "
                       "(blank = set not applicable to that scheme)")
    fig.savefig(os.path.join(FIGD, "figX1_extended_sets.png"))
    plt.close(fig)


def figX2(res):
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8), constrained_layout=True)
    for ax, fs in zip(axes, ["Spectral", "All_parameters"]):
        d = res[(res.Target == "Grain_Yield_t_ha") & (res.Feature_set == fs)]
        if not len(d):
            continue
        mat = d.pivot(index="Model", columns="Scheme", values="R2_mean")
        mat = mat.reindex(index=MODELS, columns=[s for s in SCHEMES if s in mat.columns])
        mat.columns = [LAB.get(c, c) for c in mat.columns]
        heat(ax, mat, fig, "%s — grain yield $R^2$ per model" % fs, vmin=-0.35)
    fig.suptitle("Extended sweep: individual model performance", fontsize=12,
                 fontweight="bold")
    fig.savefig(os.path.join(FIGD, "figX2_extended_models.png"))
    plt.close(fig)


def figX3(res):
    targets = ["Grain_Yield_t_ha", "Biological_Yield_t_ha", "Harvest_Index"]
    best = (res[res.Model != "Dummy_mean"].sort_values("R2_mean", ascending=False)
            .groupby(["Target", "Feature_set", "Scheme"]).first().reset_index())
    sets = [s for s in SETS if s in set(best.Feature_set)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True, constrained_layout=True)
    x = np.arange(len(sets))
    for ax, tgt in zip(axes, targets):
        for off, (sch, col, lab) in enumerate([("CV_2024-25", C_S1, S1),
                                               ("CV_2025-26", C_S2, S2)]):
            sub = best[(best.Target == tgt) & (best.Scheme == sch)] \
                .set_index("Feature_set").reindex(sets)
            ax.bar(x + (off - .5) * .4, sub.R2_mean.clip(lower=-0.15), .4, color=col, label=lab)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sets, rotation=40, ha="right", fontsize=7.5)
        ax.set_title(tgt.replace("_", " "), fontsize=10, fontweight="bold")
        ax.set_ylim(-0.2, 1.0)
    axes[0].set_ylabel("$R^2$ of best model")
    axes[0].legend(frameon=False)
    fig.suptitle("Extended sweep: every predictor set, every yield target, within season",
                 fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(FIGD, "figX3_extended_targets.png"))
    plt.close(fig)


def main():
    res = pd.read_csv(os.path.join(RESD, "ml_results_extended.csv"))
    res = res[res.R2_mean.notna()]
    for fn in (figX1, figX2, figX3):
        print("  ", fn.__name__, flush=True)
        fn(res)
    print("figures ->", FIGD)


if __name__ == "__main__":
    main()
