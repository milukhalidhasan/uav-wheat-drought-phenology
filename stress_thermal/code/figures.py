"""Figures for the drought-classifier and thermal analyses (reads results/*.csv)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, RESULTS

# validated categorical slots (dataviz reference palette, light mode)
IRR, DRO, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def fig1_classifier():
    res = pd.read_csv(RESULTS / "A1_classifier_all_configurations.csv")
    order_fs = ["Double-logistic parameters (6 VIs x 9)", "Raw trajectories, 6 VIs (20-80 DAS)",
                "Raw trajectories, all bands + VIs (20-80 DAS)", "Single date, 80 DAS (all bands + VIs)"]
    short = {order_fs[0]: "Double-logistic\nparameters", order_fs[1]: "Raw trajectories\n(6 VIs)",
             order_fs[2]: "Raw trajectories\n(all bands + VIs)", order_fs[3]: "Single flight\n(80 DAS)"}
    schemes = [("LORO", "Within season\n(leave-one-rep-out)"), ("Tr", "Other season,\nraw"),
               ("TrZ", "Other season,\nstandardised"), ("TrRef", "Other season,\n12-plot reference")]
    colors = [IRR, DRO, C3, C4]
    m = res.groupby(["feature_set", "scheme", "direction"]).balanced_acc.mean().reset_index()
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    w = 0.19
    for j, fs in enumerate(order_fs):
        for i, (sc, _) in enumerate(schemes):
            v = m[(m.feature_set == fs) & (m.scheme == sc)].balanced_acc.to_numpy()
            x = i + (j - 1.5) * w
            ax.bar(x, v.mean(), w * 0.9, color=colors[j], label=short[fs].replace("\n", " ") if i == 0 else None)
            ax.scatter([x] * len(v), v, s=10, color=INK, zorder=3, linewidths=0)
    ax.axhline(0.5, color=INK2, lw=1, ls="--")
    ax.text(3.55, 0.51, "chance", color=INK2, fontsize=8, ha="right", va="bottom")
    ax.set_xticks(range(len(schemes)), [s[1] for s in schemes])
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("Balanced accuracy\n(mean of 5 classifiers)")
    ax.set_title("Drought vs irrigated plots: within and between seasons", loc="left", fontsize=10, color=INK)
    ax.legend(ncol=2, loc="upper left", fontsize=8, bbox_to_anchor=(0, -0.2))
    ax.text(0, 1.01, "Bars: mean over both seasons or both transfer directions. Dots: each season / direction.",
            transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")
    ax.set_title("Drought vs irrigated plots: within and between seasons", loc="left", fontsize=10, color=INK, pad=16)
    save(fig, "Fig1_drought_classification_cross_season")


def fig2_senescence():
    s = pd.read_csv(RESULTS / "A7_senescence_shift_by_regime.csv")
    s = s[s.parameter == "t2_mid_senescence"]
    imp = pd.read_csv(RESULTS / "A5_dl_importance_by_parameter.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.2), gridspec_kw={"width_ratios": [1.1, 1]})
    idx = ["NDVI", "gNDVI", "GCI", "RVI", "NDRE", "CIrededge"]
    y = np.arange(len(idx))
    for k, (season, col, off) in enumerate([("2024-25", IRR, -0.12), ("2025-26", DRO, 0.12)]):
        v = s[s.Season == season].set_index("index").loc[idx, "D_minus_I"]
        a.scatter(v, y + off, s=36, color=col, label=season, zorder=3)
    a.axvline(0, color=INK2, lw=1)
    a.set_yticks(y, ["CI red-edge" if i == "CIrededge" else i for i in idx])
    a.set_xlabel("Drought minus irrigated (days)")
    a.set_title("Mid-senescence (t2) comes earlier under drought", loc="left", fontsize=10, color=INK)
    a.legend(loc="upper left", fontsize=8)
    a.set_xlim(-13, 2)
    par_names = {"t2_mid_senescence": "Mid-senescence (t2)", "Green_Duration": "Green duration",
                 "Baseline": "Baseline", "k2_rate_senescence": "Senescence rate (k2)", "Amplitude": "Amplitude",
                 "t1_mid_greenup": "Mid-green-up (t1)", "AUC_common_19_83": "Area under curve", "Peak_value": "Peak value",
                 "k1_rate_greenup": "Green-up rate (k1)"}
    p = imp.pivot(index="parameter", columns="direction", values="share_of_positive_importance_pct").fillna(0)
    p = p.loc[p.mean(axis=1).sort_values().index]
    yy = np.arange(len(p))
    for k, (d, col, off) in enumerate(zip(p.columns, [IRR, DRO], [-0.18, 0.18])):
        b.barh(yy + off, p[d], 0.34, color=col, label=d.replace(" -> ", " → "))
    b.set_yticks(yy, [par_names.get(i, i) for i in p.index])
    b.set_xlabel("Share of permutation importance (%)")
    b.set_title("What the transferred classifier uses", loc="left", fontsize=10, color=INK)
    b.legend(loc="lower right", fontsize=8, title="train → test", title_fontsize=8)
    fig.tight_layout()
    save(fig, "Fig2_senescence_shift_and_importance")


def fig3_thermal_timeseries():
    t = pd.read_csv(RESULTS / "B0_thermal_plot_by_date.csv")
    b1 = pd.read_csv(RESULTS / "B1_thermal_by_date_splitplot.csv")
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    for reg, col, lab in [("I", IRR, "Irrigated"), ("D", DRO, "Managed drought")]:
        g = t[t.Water_Regime == reg].groupby("DAS").Thermal
        mu, se = g.mean(), g.std() / np.sqrt(g.count())
        a.plot(mu.index, mu, color=col, lw=2, marker="o", ms=5, label=lab)
        a.fill_between(mu.index, mu - 1.96 * se, mu + 1.96 * se, color=col, alpha=0.15, lw=0)
    a.set_ylabel("Canopy temperature (°C)")
    a.set_title("UAV thermal canopy temperature, 2025-26 (96 plots, 10 flights)", loc="left", fontsize=10, color=INK)
    a.legend(loc="lower right", fontsize=8)
    b.bar(b1.DAS, b1.dT_D_minus_I, 3.2, color=[DRO if q < 0.05 else GRID for q in b1.Water_q], edgecolor="none")
    b.errorbar(b1.DAS, b1.dT_D_minus_I, yerr=[b1.dT_D_minus_I - b1.dT_ci_low, b1.dT_ci_high - b1.dT_D_minus_I],
               fmt="none", ecolor=INK2, lw=1, capsize=2)
    for _, r in b1.iterrows():
        if r.Water_q < 0.05:
            b.text(r.DAS, r.dT_ci_high + 0.15, f"+{r.dT_D_minus_I:.1f}", ha="center", fontsize=7.5, color=INK)
    b.axhline(0, color=INK2, lw=1)
    b.set_ylabel("Drought − irrigated (°C)")
    b.set_xlabel("Days after sowing")
    b.set_ylim(-3.5, 4.8)
    fig.tight_layout()
    fig.text(0.01, -0.01, "Orange bars: significant under split-plot error (Benjamini-Hochberg q < 0.05); "
             "error bars: 95% CI over 4 replications", fontsize=7.5, color=INK2, va="top")
    save(fig, "Fig3_thermal_timeseries_drought_vs_irrigated")


def fig4_detection_by_date():
    b5 = pd.read_csv(RESULTS / "B5_classification_thermal_vs_multispectral.csv")
    d = b5[b5.DAS != "all dates"].copy()
    d["DAS"] = d.DAS.astype(int)
    m = d.groupby(["DAS", "feature_set"]).balanced_acc.mean().unstack()
    alld = b5[b5.DAS == "all dates"].groupby("feature_set").balanced_acc.mean()
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for fs, col in [("Thermal only", DRO), ("Multispectral", IRR), ("Multispectral + thermal", C3)]:
        ax.plot(m.index, m[fs], color=col, lw=2, marker="o", ms=5, label=f"{fs}  (all dates: {alld[fs]:.2f})")
    ax.axhline(0.5, color=INK2, lw=1, ls="--")
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("Days after sowing (single flight)")
    ax.set_ylabel("Balanced accuracy\n(leave-one-rep-out)")
    ax.set_title("Detecting drought from one flight, 2025-26", loc="left", fontsize=10, color=INK)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "Fig4_drought_detection_by_flight_date")


def fig5_cwsi():
    t = pd.read_csv(RESULTS / "B0_thermal_plot_by_date.csv")
    s = t.groupby(["Plot_UID", "Water_Regime"]).agg(CWSI_e=("CWSI_e", "mean"), gsw=("gsw", "first"),
                                                     y=("Grain_Yield_t_ha", "first")).reset_index()
    c = pd.read_csv(RESULTS / "B3_season_mean_vs_physiology_yield.csv").set_index(["predictor", "target"])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    for ax, tgt, lab in [(axes[0], "y", "Grain yield (t ha⁻¹)"), (axes[1], "gsw", "Stomatal conductance, gsw (mol m⁻² s⁻¹)")]:
        for reg, col, name in [("I", IRR, "Irrigated"), ("D", DRO, "Managed drought")]:
            g = s[s.Water_Regime == reg]
            ax.scatter(g.CWSI_e, g[tgt], s=16, color=col, alpha=0.85, label=name, edgecolors="white", linewidths=0.5)
        key = "Grain_Yield_t_ha" if tgt == "y" else "gsw"
        r = c.loc[("season mean CWSI_e", key)]
        ax.text(0.98, 0.97, f"r = {r.r_pooled:.2f} (all plots)\nr = {r.r_within_regime:.2f} (within regime)",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)
        ax.set_xlabel("Season-mean empirical CWSI (UAV thermal)")
        ax.set_ylabel(lab)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Thermal stress index vs yield and leaf physiology, 2025-26", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, "Fig5_CWSI_vs_yield_and_gsw")


def fig6_corr_by_date():
    c = pd.read_csv(RESULTS / "B3_B4_correlations_by_date.csv")
    c = c[c.target == "Grain_Yield_t_ha"]
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    for pred, col in [("Thermal", DRO), ("NDVI", IRR)]:
        g = c[c.predictor == pred].sort_values("DAS")
        ax.plot(g.DAS, g.r_pooled, color=col, lw=2, marker="o", ms=5, label=f"{pred}, all plots")
        ax.plot(g.DAS, g.r_within_regime, color=col, lw=1.5, ls="--", marker="o", ms=4, mfc="white",
                label=f"{pred}, within water regime")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("Days after sowing")
    ax.set_ylabel("Pearson r with grain yield")
    ax.set_title("When does each sensor carry yield information? (2025-26)", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7.5, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.16))
    ax.set_ylim(-0.9, 1.0)
    save(fig, "Fig6_correlation_with_yield_by_date")


def figS1_chm():
    d = pd.read_csv(RESULTS / "C_chm_height_pairs.csv")
    r = pd.read_csv(RESULTS / "C_chm_vs_plant_height.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for reg, col, name in [("I", IRR, "Irrigated"), ("D", DRO, "Managed drought")]:
        g = d[d.Water_Regime == reg]
        ax.scatter(g.Plant_height_cm, g.dchm, s=8, color=col, alpha=0.6, label=name, linewidths=0)
    pooled = r[r.DAS == "all dates pooled"].iloc[0]
    ax.text(0.02, 0.97, f"pooled r = {pooled.r:.2f}; per-date r = "
            f"{r[r.DAS != 'all dates pooled'].r.astype(float).min():.2f} to {r[r.DAS != 'all dates pooled'].r.astype(float).max():.2f}",
            transform=ax.transAxes, va="top", fontsize=7.5, color=INK)
    ax.set_xlabel("Ruler plant height (cm)")
    ax.set_ylabel("UAV canopy height, dCHM")
    ax.set_title("Supplementary: UAV canopy height check, 2025-26", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, "FigS1_CHM_vs_plant_height")


if __name__ == "__main__":
    fig1_classifier()
    fig2_senescence()
    fig3_thermal_timeseries()
    fig4_detection_by_date()
    fig5_cwsi()
    fig6_corr_by_date()
    figS1_chm()
    print("figures written to", FIGURES)
