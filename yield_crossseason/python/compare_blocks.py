"""
The four comparisons in the analysis plan, laid out explicitly.

  1. within 2024-25 : agronomic vs double-logistic vs hybrid
  2. within 2025-26 : the same
  3. 2024-25 vs 2025-26 : is the ranking stable between seasons?
  4. across seasons : pooled and transfer, for each block and the hybrid

Reads the core, extended and hybrid sweeps, merges them, and writes:
  results/block_comparison.csv          tidy long form of every comparison
  results/block_comparison_wide.csv     the headline matrix
  figures/figH1_block_comparison.png    per-season head-to-head
  figures/figH2_hybrid_gain.png         what the hybrid adds over its parts
  figures/figH3_season_ranking.png      ranking stability between seasons
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wheat_paths import RESD, FIGD

S1, S2 = "2024-25", "2025-26"
BLOCKS = ["Design_only", "Agro_predictive", "Pheno", "Agro+Pheno",
          "Spectral", "Agro+Spectral", "Pheno+Spectral", "All_parameters"]
NICE = {"Design_only": "Treatment only", "Agro_predictive": "Agronomic",
        "Pheno": "Double-logistic", "Agro+Pheno": "HYBRID  Agro + Double-logistic",
        "Spectral": "Spectral", "Agro+Spectral": "Agro + Spectral",
        "Pheno+Spectral": "Double-logistic + Spectral",
        "All_parameters": "Everything"}
SCHEMES = ["CV_2024-25", "Block_2024-25", "CV_2025-26", "Block_2025-26", "CV_pooled",
           "Transfer_24to25", "Transfer_25to24", "TransferZ_24to25", "TransferZ_25to24"]
LAB = {"CV_2024-25": "CV 24-25", "Block_2024-25": "Blk 24-25", "CV_2025-26": "CV 25-26",
       "Block_2025-26": "Blk 25-26", "CV_pooled": "Pooled",
       "Transfer_24to25": "Tr 24>25", "Transfer_25to24": "Tr 25>24",
       "TransferZ_24to25": "TrZ 24>25", "TransferZ_25to24": "TrZ 25>24"}

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "figure.facecolor": "white", "savefig.bbox": "tight"})
C_S1, C_S2 = "#2B6CB0", "#C05621"


def load():
    parts = []
    for f in ("ml_results_extended.csv", "ml_results_hybrid.csv", "ml_results_all.csv"):
        p = os.path.join(RESD, f)
        if os.path.exists(p):
            d = pd.read_csv(p)
            d["source"] = f
            parts.append(d)
    res = pd.concat(parts, ignore_index=True)
    res = res[res.R2_mean.notna()]
    # the extended sweep supersedes the core one where both ran a set
    res["prio"] = res.source.map({"ml_results_hybrid.csv": 0,
                                  "ml_results_extended.csv": 1,
                                  "ml_results_all.csv": 2})
    res = (res.sort_values("prio")
              .drop_duplicates(["Target", "Feature_set", "Model", "Scheme"], keep="first"))
    return res


def best(res, target):
    d = res[(res.Target == target) & (res.Model != "Dummy_mean")]
    return (d.sort_values("R2_mean", ascending=False)
             .groupby(["Feature_set", "Scheme"]).first().reset_index())


def fig_head_to_head(b, target):
    sets = [s for s in BLOCKS if s in set(b.Feature_set)]
    fig, ax = plt.subplots(figsize=(9.5, 4.6), constrained_layout=True)
    x = np.arange(len(sets))
    for off, (sch, col, lab) in enumerate([("CV_2024-25", C_S1, S1),
                                           ("CV_2025-26", C_S2, S2)]):
        sub = b[b.Scheme == sch].set_index("Feature_set").reindex(sets)
        bars = ax.bar(x + (off - .5) * .4, sub.R2_mean, .4, color=col, label=lab)
        for r, v, m in zip(bars, sub.R2_mean, sub.Model):
            if np.isfinite(v):
                ax.text(r.get_x() + r.get_width() / 2, v + .012, "%.2f" % v,
                        ha="center", fontsize=7.5)
                ax.text(r.get_x() + r.get_width() / 2, .02, m, ha="center", va="bottom",
                        fontsize=6, rotation=90, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([NICE.get(s, s) for s in sets], rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("$R^2$ of best model")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    ax.set_title("Predictor blocks head to head, within season — %s"
                 % target.replace("_", " "), fontsize=11, fontweight="bold", loc="left")
    fig.savefig(os.path.join(FIGD, "figH1_block_comparison.png"))
    plt.close(fig)


def fig_hybrid_gain(b):
    """What does combining agronomic + double-logistic actually buy?"""
    rows = []
    for sch in SCHEMES:
        g = b[b.Scheme == sch].set_index("Feature_set").R2_mean
        if not {"Agro_predictive", "Pheno", "Agro+Pheno"} <= set(g.index):
            continue
        rows.append({"Scheme": LAB.get(sch, sch), "Agronomic": g["Agro_predictive"],
                     "Double-logistic": g["Pheno"], "Hybrid": g["Agro+Pheno"],
                     "best_single": max(g["Agro_predictive"], g["Pheno"])})
    d = pd.DataFrame(rows)
    d["gain"] = d.Hybrid - d.best_single
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
    x = np.arange(len(d))
    for i, (c, col) in enumerate([("Agronomic", "#718096"),
                                  ("Double-logistic", "#38A169"),
                                  ("Hybrid", "#2B6CB0")]):
        axes[0].bar(x + (i - 1) * .27, d[c], .27, color=col, label=c)
    axes[0].axhline(0, color="black", lw=.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(d.Scheme, rotation=30, ha="right")
    axes[0].set_ylabel("$R^2$"); axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Agronomic vs double-logistic vs their hybrid",
                      fontsize=10, fontweight="bold", loc="left")

    cols = ["#2F855A" if v > 0 else "#C53030" for v in d.gain]
    axes[1].bar(x, d.gain, .55, color=cols)
    for xi, v in zip(x, d.gain):
        axes[1].text(xi, v + (.006 if v >= 0 else -.02), "%+.3f" % v, ha="center",
                     fontsize=7.5)
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(d.Scheme, rotation=30, ha="right")
    axes[1].set_ylabel("$R^2$ gain of hybrid over its better half")
    axes[1].set_title("Does combining them help?", fontsize=10, fontweight="bold", loc="left")
    fig.savefig(os.path.join(FIGD, "figH2_hybrid_gain.png"))
    plt.close(fig)
    return d


def fig_season_ranking(b):
    sets = [s for s in BLOCKS if s in set(b.Feature_set)]
    a = b[b.Scheme == "CV_2024-25"].set_index("Feature_set").reindex(sets).R2_mean
    c = b[b.Scheme == "CV_2025-26"].set_index("Feature_set").reindex(sets).R2_mean
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    for s in sets:
        ax.plot([0, 1], [a[s], c[s]], "-o", lw=1.8, ms=6)
        ax.annotate(NICE.get(s, s), (1.02, c[s]), fontsize=8, va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels([S1, S2])
    ax.set_xlim(-0.08, 1.75)
    ax.set_ylabel("$R^2$ of best model, grain yield")
    ax.set_title("Is the block ranking stable between seasons?", fontsize=11,
                 fontweight="bold", loc="left")
    fig.savefig(os.path.join(FIGD, "figH3_season_ranking.png"))
    plt.close(fig)


def main():
    res = load()
    have = sorted(set(res.Feature_set) & set(BLOCKS))
    print("blocks available:", have)

    out = []
    for target in ["Grain_Yield_t_ha", "Biological_Yield_t_ha", "Harvest_Index"]:
        b = best(res, target)
        b = b[b.Feature_set.isin(BLOCKS)]
        b["Target"] = target
        out.append(b[["Target", "Feature_set", "Scheme", "Model", "n_features",
                      "R2_mean", "R2_sd", "RMSE_mean", "bias_mean"]])
    tidy = pd.concat(out, ignore_index=True)
    tidy.to_csv(os.path.join(RESD, "block_comparison.csv"), index=False)

    gy = best(res, "Grain_Yield_t_ha")
    gy = gy[gy.Feature_set.isin(BLOCKS)]
    wide = gy.pivot(index="Feature_set", columns="Scheme", values="R2_mean")
    wide = wide.reindex([s for s in BLOCKS if s in wide.index],
                        columns=[s for s in SCHEMES if s in wide.columns])
    wide.to_csv(os.path.join(RESD, "block_comparison_wide.csv"))

    fig_head_to_head(gy, "Grain_Yield_t_ha")
    d = fig_hybrid_gain(gy)
    fig_season_ranking(gy)

    print("\nGRAIN YIELD, best model R2 by block and scheme")
    print(wide.round(3).to_string())
    print("\nHYBRID vs its parts")
    print(d.round(3).to_string(index=False))
    print("\nwrote block_comparison.csv / block_comparison_wide.csv and figH1-figH3")


if __name__ == "__main__":
    main()
