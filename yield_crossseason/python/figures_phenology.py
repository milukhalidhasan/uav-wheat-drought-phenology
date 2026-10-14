"""
Publication figures for the double-logistic phenology and the underlying data.

  figP1  fitted double-logistic curves by water regime, both seasons, 6 indices
  figP2  fitted NDVI curves by nitrogen level, split by water regime and season
  figP3  distribution of every curve parameter, season x water regime
  figP4  derived phenological stages on a calendar of days after sowing
  figP5  curve fit quality (R2, RMSE, parameters at bound)
  figP6  AUC vs grain yield, per index, per season, with fitted line
  figP7  raw vegetation index trajectories (observed means +/- SE)
  figP8  ground measurements over time by water regime
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from wheat_paths import (ROOT, F24, F25, OUT, CSVD, XLSX, RESD, FIGD,
                         ML_DIR, REPORT_DOCX, check_inputs)

S1, S2 = "2024-25", "2025-26"
INDICES = ["NDVI", "NDRE", "CI(rededge)", "GCI", "gNDVI", "RVI"]
WINDOW = {S1: (10, 108), S2: (19, 83)}

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "figure.facecolor": "white", "savefig.bbox": "tight"})
C_D, C_I = "#C05621", "#2B6CB0"          # drought, irrigated
C_S1, C_S2 = "#2B6CB0", "#C05621"        # season 1, season 2
N_COLORS = ["#CBD5E0", "#90CDF4", "#4299E1", "#2A4365"]

rd = lambda n: pd.read_csv(os.path.join(CSVD, n + ".csv"))


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def curve(t, base, amp, t1, t2, k1, k2):
    return base + amp * (sig(k1 * (t - t1)) - sig(k2 * (t - t2)))


def eval_rows(sub, t):
    """Mean fitted curve over a set of plots."""
    ys = np.array([curve(t, r.Baseline, r.Amplitude, r.t1_mid_greenup,
                         r.t2_mid_senescence, r.k1_rate_greenup, r.k2_rate_senescence)
                   for r in sub.itertuples()])
    return ys.mean(axis=0), ys.std(axis=0) / np.sqrt(max(len(ys), 1))


# ---------------------------------------------------------------------------
def figP1(ph):
    fig, axes = plt.subplots(2, len(INDICES), figsize=(17, 6.4), constrained_layout=True)
    for i, season in enumerate([S1, S2]):
        lo, hi = WINDOW[season]
        t = np.linspace(lo, hi, 300)
        for j, idx in enumerate(INDICES):
            ax = axes[i, j]
            for wr, col, lab in [("D", C_D, "Drought"), ("I", C_I, "Irrigated")]:
                sub = ph[(ph.Season == season) & (ph.Index == idx) & (ph.Water_Regime == wr)]
                m, se = eval_rows(sub, t)
                ax.plot(t, m, color=col, lw=2, label=lab)
                ax.fill_between(t, m - 1.96 * se, m + 1.96 * se, color=col, alpha=0.2, lw=0)
            if i == 0:
                ax.set_title(idx, fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel("%s\nindex value" % season, fontsize=9, fontweight="bold")
            ax.set_xlabel("Days after sowing")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fitted double-logistic canopy trajectories by water regime "
                 "(mean $\\pm$ 95% CI of plot-level fits)", fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(FIGD, "figP1_curves_by_water.png"))
    plt.close(fig)


def figP2(ph):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for i, season in enumerate([S1, S2]):
        lo, hi = WINDOW[season]
        t = np.linspace(lo, hi, 300)
        for j, (wr, wlab) in enumerate([("D", "Drought"), ("I", "Irrigated")]):
            ax = axes[i, j]
            for k, nlev in enumerate(["N0", "N1", "N2", "N3"]):
                sub = ph[(ph.Season == season) & (ph.Index == "NDVI") &
                         (ph.Water_Regime == wr) & (ph.N_Level == nlev)]
                if not len(sub):
                    continue
                m, _ = eval_rows(sub, t)
                ax.plot(t, m, color=N_COLORS[k], lw=2,
                        label="%s (%s kg/ha)" % (nlev, [0, 210, 300, 390][k]))
            ax.set_title("%s — %s" % (season, wlab), fontsize=9, fontweight="bold")
            ax.set_xlabel("Days after sowing")
            ax.set_ylabel("NDVI")
    axes[0, 0].legend(frameon=False, fontsize=7.5, title="Nitrogen", title_fontsize=8)
    fig.suptitle("Fitted NDVI trajectories by nitrogen level", fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(FIGD, "figP2_curves_by_nitrogen.png"))
    plt.close(fig)


def figP3(ph):
    pars = ["Baseline", "Amplitude", "Peak_value", "t1_mid_greenup", "t2_mid_senescence",
            "k1_rate_greenup", "k2_rate_senescence", "Green_Duration", "LOS",
            "AUC_common_19_83", "Rate_greenup", "Rate_senescence"]
    nd = ph[ph.Index == "NDVI"]
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), constrained_layout=True)
    for ax, p in zip(axes.ravel(), pars):
        data, labels, colors = [], [], []
        for season, base in [(S1, C_S1), (S2, C_S2)]:
            for wr, sh in [("D", 0.55), ("I", 1.0)]:
                v = nd[(nd.Season == season) & (nd.Water_Regime == wr)][p].dropna()
                data.append(v.values)
                labels.append("%s\n%s" % (season[-5:], wr))
                colors.append(base if sh == 1.0 else base + "99")
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)
            patch.set_edgecolor("black")
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(p, fontsize=9, fontweight="bold")
    fig.suptitle("Double-logistic parameter distributions (NDVI), season $\\times$ water regime",
                 fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(FIGD, "figP3_parameter_distributions.png"))
    plt.close(fig)


def figP4(ph):
    stages = ["SOS", "EOG", "POS", "SOD", "EOS"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True, sharex=True)
    for ax, season in zip(axes, [S1, S2]):
        y = 0
        ticks, labels = [], []
        for idx in INDICES:
            for wr, col in [("D", C_D), ("I", C_I)]:
                sub = ph[(ph.Season == season) & (ph.Index == idx) & (ph.Water_Regime == wr)]
                vals = [sub[s].mean() for s in stages]
                errs = [sub[s].std() for s in stages]
                ax.plot(vals, [y] * len(vals), "-", color=col, lw=1.2, alpha=0.6)
                ax.errorbar(vals, [y] * len(vals), xerr=errs, fmt="o", color=col,
                            ms=5, capsize=2, lw=0.8)
                ticks.append(y)
                labels.append("%s %s" % (idx, wr))
                y += 1
            y += 0.5
        lo, hi = WINDOW[season]
        ax.axvspan(lo, hi, color="#68D391", alpha=0.10, lw=0)
        ax.axvline(lo, color="grey", ls=":", lw=1)
        ax.axvline(hi, color="grey", ls=":", lw=1)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("Days after sowing")
        ax.set_title("%s   (shaded = UAV flight window)" % season,
                     fontsize=10, fontweight="bold")
    fig.suptitle("Derived phenological stages: SOS, EOG, POS, SOD, EOS "
                 "(mean $\\pm$ SD across plots)", fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(FIGD, "figP4_stages.png"))
    plt.close(fig)


def figP5(ph):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    x = np.arange(len(INDICES))
    for off, (season, col) in enumerate([(S1, C_S1), (S2, C_S2)]):
        d = ph[ph.Season == season]
        r2 = [d[d.Index == i].R2.mean() for i in INDICES]
        r2e = [d[d.Index == i].R2.std() for i in INDICES]
        rm = [d[d.Index == i].RMSE.mean() for i in INDICES]
        bd = [(d[d.Index == i].Params_At_Bound > 0).mean() * 100 for i in INDICES]
        axes[0].bar(x + (off - .5) * .4, r2, .4, yerr=r2e, color=col, label=season, capsize=2)
        axes[1].bar(x + (off - .5) * .4, rm, .4, color=col, label=season)
        axes[2].bar(x + (off - .5) * .4, bd, .4, color=col, label=season)
    for ax, ttl, yl in zip(axes,
                           ["Curve fit $R^2$", "Curve fit RMSE", "Curves with a parameter at a bound"],
                           ["$R^2$", "RMSE (index units)", "% of plots"]):
        ax.set_xticks(x)
        ax.set_xticklabels(INDICES, rotation=30, ha="right")
        ax.set_title(ttl, fontsize=10, fontweight="bold")
        ax.set_ylabel(yl)
    axes[0].set_ylim(0.5, 1.02)
    axes[0].legend(frameon=False)
    fig.suptitle("Double-logistic fit quality by index and season", fontsize=12,
                 fontweight="bold")
    fig.savefig(os.path.join(FIGD, "figP5_fit_quality.png"))
    plt.close(fig)


def figP6(ph, harvest):
    gy = harvest.set_index("Plot_UID").Grain_Yield_t_ha
    fig, axes = plt.subplots(2, len(INDICES), figsize=(17, 6.2), constrained_layout=True)
    for i, season in enumerate([S1, S2]):
        for j, idx in enumerate(INDICES):
            ax = axes[i, j]
            sub = ph[(ph.Season == season) & (ph.Index == idx)].copy()
            sub["GY"] = sub.Plot_UID.map(gy)
            for wr, col in [("D", C_D), ("I", C_I)]:
                s = sub[sub.Water_Regime == wr]
                ax.scatter(s.AUC_common_19_83, s.GY, s=16, color=col, alpha=0.7,
                           edgecolor="white", linewidth=0.3,
                           label="Drought" if wr == "D" else "Irrigated")
            ok = sub[["AUC_common_19_83", "GY"]].dropna()
            if len(ok) > 3:
                sl, ic, r, p, _ = stats.linregress(ok.AUC_common_19_83, ok.GY)
                xs = np.linspace(ok.AUC_common_19_83.min(), ok.AUC_common_19_83.max(), 50)
                ax.plot(xs, ic + sl * xs, "k-", lw=1.4)
                ax.text(0.04, 0.95, "r = %.2f" % r, transform=ax.transAxes,
                        va="top", fontsize=8.5, fontweight="bold")
            if i == 0:
                ax.set_title(idx, fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel("%s\nGrain yield (t/ha)" % season, fontsize=9, fontweight="bold")
            ax.set_xlabel("AUC (19-83 DAS)")
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("Cross-season-comparable AUC vs grain yield, by index",
                 fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(FIGD, "figP6_auc_vs_yield.png"))
    plt.close(fig)


def figP7(spec):
    fig, axes = plt.subplots(2, len(INDICES), figsize=(17, 6.2), constrained_layout=True)
    for i, season in enumerate([S1, S2]):
        for j, idx in enumerate(INDICES):
            ax = axes[i, j]
            d = spec[spec.Season == season]
            for wr, col in [("D", C_D), ("I", C_I)]:
                g = d[d.Water_Regime == wr].groupby("DAS")[idx].agg(["mean", "sem"])
                ax.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], color=col,
                            marker="o", ms=4, lw=1.6, capsize=2,
                            label="Drought" if wr == "D" else "Irrigated")
            if i == 0:
                ax.set_title(idx, fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel("%s\nobserved index" % season, fontsize=9, fontweight="bold")
            ax.set_xlabel("Days after sowing")
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Observed vegetation index trajectories (mean $\\pm$ 95% CI, not fitted)",
                 fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(FIGD, "figP7_observed_trajectories.png"))
    plt.close(fig)


def figP8(ground):
    vars_ = ["Chlorophyll", "Plant_height_cm", "Soil_Temp_C", "Soil_Moisture_pct"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    for i, season in enumerate([S1, S2]):
        d = ground[ground.Season == season]
        for j, v in enumerate(vars_):
            ax = axes[i, j]
            if d[v].notna().sum() == 0:
                ax.text(0.5, 0.5, "not recorded\nin %s" % season, ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="grey")
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            else:
                for wr, col in [("D", C_D), ("I", C_I)]:
                    g = d[d.Water_Regime == wr].groupby("DAS")[v].agg(["mean", "sem"]).dropna()
                    ax.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], color=col,
                                marker="o", ms=4, lw=1.6, capsize=2,
                                label="Drought" if wr == "D" else "Irrigated")
                ax.set_xlabel("Days after sowing")
            if i == 0:
                ax.set_title(v, fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(season, fontsize=9, fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Ground measurements over time by water regime", fontsize=12,
                 fontweight="bold")
    fig.savefig(os.path.join(FIGD, "figP8_ground_timeseries.png"))
    plt.close(fig)


def main():
    os.makedirs(FIGD, exist_ok=True)
    ph = rd("Phenology_Parameters")
    design = rd("Plot_Design")[["Plot_UID", "N_Level", "Seaweed_Level", "N_kg_ha"]]
    ph = ph.merge(design, on="Plot_UID", how="left")
    harvest = rd("Harvest_PlotLevel")
    spec = rd("Spectral_TimeSeries")
    ground = rd("Ground_TimeSeries")

    for fn, args in [(figP1, (ph,)), (figP2, (ph,)), (figP3, (ph,)), (figP4, (ph,)),
                     (figP5, (ph,)), (figP6, (ph, harvest)), (figP7, (spec,)),
                     (figP8, (ground,))]:
        print("  ", fn.__name__, flush=True)
        fn(*args)
    print("figures ->", FIGD)


if __name__ == "__main__":
    main()
