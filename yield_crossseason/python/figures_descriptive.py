"""
Descriptive and treatment figures.

  figD1  yield and yield components by water regime x nitrogen x seaweed
  figD2  when to fly: correlation of each index with grain yield against DAS
  figD3  top univariate predictors of grain yield, both seasons side by side
  figD4  correlation structure among the headline parameters
  figD5  season comparison of every measured trait (effect size)
  figD6  treatment effect sizes: which parameters respond to water / N / seaweed
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import extended_features as E

from wheat_paths import (ROOT, F24, F25, OUT, CSVD, XLSX, RESD, FIGD,
                         ML_DIR, REPORT_DOCX, check_inputs)

CSVD = E.CSVD
S1, S2 = E.S1, E.S2
INDICES = ["NDVI", "NDRE", "CI(rededge)", "GCI", "gNDVI", "RVI"]
DAS_GRID = [20, 30, 40, 50, 60, 70, 80]

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "figure.facecolor": "white", "savefig.bbox": "tight"})
C_D, C_I = "#C05621", "#2B6CB0"
C_S1, C_S2 = "#2B6CB0", "#C05621"
rd = lambda n: pd.read_csv(os.path.join(CSVD, n + ".csv"))


def figD1(df):
    traits = ["Grain_Yield_t_ha", "Biological_Yield_t_ha", "Harvest_Index",
              "Total_Grain_per_Plant", "TSW_g", "Spike_Length_cm"]
    fig, axes = plt.subplots(2, len(traits), figsize=(17, 6.4), constrained_layout=True)
    for i, season in enumerate([S1, S2]):
        d = df[df.Season == season]
        for j, tr in enumerate(traits):
            ax = axes[i, j]
            for wr, col in [("D", C_D), ("I", C_I)]:
                g = d[d.Water_Regime == wr].groupby("N_kg_ha")[tr].agg(["mean", "sem"])
                ax.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], color=col,
                            marker="o", ms=5, lw=1.8, capsize=3,
                            label="Drought" if wr == "D" else "Irrigated")
            if i == 0:
                ax.set_title(tr.replace("_", " "), fontsize=9.5, fontweight="bold")
            if j == 0:
                ax.set_ylabel(season, fontsize=9, fontweight="bold")
            ax.set_xlabel("N (kg/ha)")
            ax.set_xticks([0, 210, 300, 390])
            ax.tick_params(axis="x", labelsize=7.5)
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Yield and components by nitrogen rate and water regime "
                 "(mean $\\pm$ 95% CI)", fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(FIGD, "figD1_treatment_response.png"))
    plt.close(fig)


def figD2(df):
    """Correlation of each index with grain yield as a function of flight date."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, season in zip(axes, [S1, S2]):
        d = df[df.Season == season]
        y = d.Grain_Yield_t_ha.to_numpy(float)
        for k, idx in enumerate(INDICES):
            rs = []
            for das in DAS_GRID:
                c = "%s_DAS%d" % (idx, das)
                if c not in d:
                    rs.append(np.nan)
                    continue
                x = d[c].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                rs.append(stats.pearsonr(x[ok], y[ok])[0] if ok.sum() > 10 else np.nan)
            ax.plot(DAS_GRID, rs, marker="o", ms=5, lw=1.8,
                    color=cmap(k / (len(INDICES) - 1)), label=idx)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Days after sowing of the flight")
        ax.set_title(season, fontsize=10, fontweight="bold")
        ax.set_xticks(DAS_GRID)
    axes[0].set_ylabel("Pearson r with grain yield")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("When to fly: index-to-yield correlation against flight date",
                 fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(FIGD, "figD2_when_to_fly.png"))
    plt.close(fig)


def figD3():
    cons = pd.read_csv(os.path.join(RESD, "univariate_consistency.csv"))
    d = cons[(cons.Target == "Grain_Yield_t_ha")].copy()
    d = d[d.consistent].sort_values("mean_abs_r", ascending=False).head(25)[::-1]
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    y = np.arange(len(d))
    ax.barh(y - 0.2, d.r_2024_25, 0.4, color=C_S1, label=S1)
    ax.barh(y + 0.2, d.r_2025_26, 0.4, color=C_S2, label=S2)
    ax.set_yticks(y)
    ax.set_yticklabels([("%s  [%s]" % (p, b)) for p, b in zip(d.Parameter, d.Block)],
                       fontsize=7.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson r with grain yield")
    ax.legend(frameon=False)
    ax.set_title("Strongest associations present in BOTH seasons\n"
                 "(significant after Benjamini-Hochberg correction in each season)",
                 fontsize=11, fontweight="bold", loc="left")
    fig.savefig(os.path.join(FIGD, "figD3_top_univariate.png"))
    plt.close(fig)


def figD4(df):
    keys = ["Grain_Yield_t_ha", "Biological_Yield_t_ha", "Harvest_Index",
            "Fresh_Biomass_t_ha", "Total_Grain_per_Plant", "TSW_g", "Tiller_No",
            "Spike_Length_cm", "Plant_Length_cm", "Root_Length_cm",
            "Chlorophyll_mean", "Plant_height_max_cm", "Soil_Temp_mean_C",
            "H2O2_umol_per_gFW", "gsw", "PhiPS2", "ETR", "Tleaf",
            "NDVI_DAS70", "NDVI_DAS80", "NDVI_AUC_common_19_83", "NDVI_Peak_value",
            "NDVI_SOS", "NDVI_EOS", "NDVI_Green_Duration"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
    for ax, season in zip(axes, [S1, S2]):
        d = df[df.Season == season]
        k = [c for c in keys if c in d.columns and d[c].notna().sum() > 20]
        M = d[k].corr()
        im = ax.imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(k)))
        ax.set_xticklabels([c.replace("_", " ") for c in k], rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(k)))
        ax.set_yticklabels([c.replace("_", " ") for c in k], fontsize=6.5)
        ax.set_title(season, fontsize=11, fontweight="bold")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.suptitle("Correlation structure among headline parameters",
                 fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(FIGD, "figD4_correlation_matrix.png"))
    plt.close(fig)


def figD5():
    sc = pd.read_csv(os.path.join(RESD, "season_comparison.csv"))
    keep = ["Grain_Yield_t_ha", "Biological_Yield_t_ha", "Harvest_Index",
            "Fresh_Biomass_t_ha", "Total_Grain_per_Plant", "TSW_g", "Tiller_No",
            "Spike_Length_cm", "Plant_Length_cm", "Root_Length_cm",
            "H2O2_umol_per_gFW", "gsw", "PhiPS2", "ETR", "Tleaf",
            "Chlorophyll_mean", "Plant_height_max_cm", "Soil_Temp_mean_C",
            "NDVI_AUC_common_19_83", "NDVI_Peak_value", "NDVI_SOS", "NDVI_EOS",
            "NDVI_Green_Duration", "NDVI_Amplitude"]
    d = sc[sc.Parameter.isin(keep)].copy().sort_values("hedges_g")
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    cols = ["#C05621" if g < 0 else "#2B6CB0" for g in d.hedges_g]
    ax.barh(range(len(d)), d.hedges_g, color=cols)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([p.replace("_", " ") for p in d.Parameter], fontsize=7.5)
    ax.axvline(0, color="black", lw=0.8)
    for x in (-0.8, -0.5, 0.5, 0.8):
        ax.axvline(x, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("Hedges' g   (positive = higher in 2024-25, negative = higher in 2025-26)")
    ax.set_title("How much each parameter shifted between seasons\n"
                 "dotted lines at |g| = 0.5 (medium) and 0.8 (large)",
                 fontsize=11, fontweight="bold", loc="left")
    fig.savefig(os.path.join(FIGD, "figD5_season_shift.png"))
    plt.close(fig)


def figD6():
    av = pd.read_csv(os.path.join(RESD, "treatment_effects.csv"))
    terms = ["Water", "N", "Seaweed", "WxN", "WxS", "NxS"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True, sharey=True)
    for ax, season in zip(axes, [S1, S2]):
        d = av[av.Season == season]
        counts = [(d["q_" + t] < 0.05).sum() if "q_" + t in d else 0 for t in terms]
        pct = [100 * c / max(len(d), 1) for c in counts]
        bars = ax.bar(terms, pct, color=["#2B6CB0", "#38A169", "#D69E2E",
                                         "#A0AEC0", "#A0AEC0", "#A0AEC0"])
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.7, str(c),
                    ha="center", fontsize=8, fontweight="bold")
        ax.set_title("%s   (n = %d parameters tested)" % (season, len(d)),
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("ANOVA term")
    axes[0].set_ylabel("% of parameters with q < 0.05")
    fig.suptitle("Which experimental factors actually moved the measured parameters?",
                 fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(FIGD, "figD6_treatment_effects.png"))
    plt.close(fig)


def main():
    os.makedirs(FIGD, exist_ok=True)
    df, blocks = E.build()
    for fn, args in [(figD1, (df,)), (figD2, (df,)), (figD3, ()), (figD4, (df,)),
                     (figD5, ()), (figD6, ())]:
        print("  ", fn.__name__, flush=True)
        fn(*args)
    print("figures ->", FIGD)


if __name__ == "__main__":
    main()
