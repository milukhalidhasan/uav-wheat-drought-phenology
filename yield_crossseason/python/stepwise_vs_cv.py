"""
How much does AIC-selected multiple linear regression overstate its own accuracy?

The established approach in this literature (Tuhin et al., 2025, Smart Agricultural
Technology) fits the double-logistic parameters a-g to grain yield with multiple
linear regression, selects a subset by AIC, and reports the resulting R2. That
number is computed on the same data used to fit AND to select, so it is an
in-sample, post-selection statistic.

This script quantifies the gap on our data. For each season and each index it:
  1. selects the best subset of the six curve parameters by AIC on all the data,
     and reports the in-sample R2 - the analogue of the published figure;
  2. repeats the whole procedure, selection included, inside every fold of a
     repeated 5-fold cross-validation, and reports the out-of-fold R2.

The difference between the two is the optimism of the published statistic.
"""
import os
import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from wheat_paths import RESD, CSVD, FIGD

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S1, S2 = "2024-25", "2025-26"
SEED = 20260919
# the six free parameters of the double logistic, in the harmonised naming
PARAMS = ["Baseline", "Amplitude", "t1_mid_greenup", "t2_mid_senescence",
          "k1_rate_greenup", "k2_rate_senescence"]
INDICES = ["NDVI", "NDRE", "CI(rededge)", "GCI", "gNDVI", "RVI"]

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "figure.facecolor": "white", "savefig.bbox": "tight"})
C_S1, C_S2 = "#2B6CB0", "#C05621"


def ols_fit(X, y):
    A = np.column_stack([np.ones(len(y)), X]) if X.shape[1] else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def ols_predict(beta, X):
    A = np.column_stack([np.ones(len(X)), X]) if X.shape[1] else np.ones((len(X), 1))
    return A @ beta


def aic(y, yhat, k):
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    rss = max(rss, 1e-12)
    return n * np.log(rss / n) + 2 * (k + 1)


def best_subset(X, y, names):
    """Exhaustive best-subset selection by AIC - a generous reading of
    'stepwise AIC', since it cannot get stuck in a local optimum."""
    best, best_aic = (), np.inf
    for r in range(len(names) + 1):
        for combo in itertools.combinations(range(len(names)), r):
            Xs = X[:, combo]
            beta = ols_fit(Xs, y)
            a = aic(y, ols_predict(beta, Xs), len(combo))
            if a < best_aic:
                best, best_aic = combo, a
    return best


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)


def run():
    ph = pd.read_csv(os.path.join(CSVD, "Phenology_Parameters.csv"))
    hv = pd.read_csv(os.path.join(CSVD, "Harvest_PlotLevel.csv"))
    gy = hv.set_index("Plot_UID").Grain_Yield_t_ha

    rows = []
    for season in (S1, S2):
        for idx in INDICES:
            d = ph[(ph.Season == season) & (ph.Index == idx)].copy()
            d["y"] = d.Plot_UID.map(gy)
            d = d.dropna(subset=PARAMS + ["y"])
            X = d[PARAMS].to_numpy(float)
            y = d["y"].to_numpy(float)
            mu, sd = X.mean(0), X.std(0)
            sd[sd == 0] = 1
            Xs = (X - mu) / sd

            # 1. select on everything, score on everything  (the published statistic)
            sel = best_subset(Xs, y, PARAMS)
            beta = ols_fit(Xs[:, sel], y)
            r2_in = r2(y, ols_predict(beta, Xs[:, sel]))
            k = len(sel)
            r2_adj = 1 - (1 - r2_in) * (len(y) - 1) / (len(y) - k - 1)

            # 2. redo the selection inside every fold  (honest)
            cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
            oof = np.full((3, len(y)), np.nan)
            for i, (tr, te) in enumerate(cv.split(Xs)):
                mtr, str_ = Xs[tr].mean(0), Xs[tr].std(0)
                str_[str_ == 0] = 1
                Xtr = (Xs[tr] - mtr) / str_
                Xte = (Xs[te] - mtr) / str_
                s = best_subset(Xtr, y[tr], PARAMS)
                b = ols_fit(Xtr[:, s], y[tr])
                oof[i // 5, te] = ols_predict(b, Xte[:, s])
            r2_cv = float(np.mean([r2(y, oof[r]) for r in range(3)]))

            rows.append({"Season": season, "Index": idx, "n": len(y),
                         "n_selected": k,
                         "selected": ", ".join(PARAMS[i] for i in sel) or "(intercept only)",
                         "R2_in_sample": round(r2_in, 3),
                         "R2_adjusted": round(r2_adj, 3),
                         "R2_cross_validated": round(r2_cv, 3),
                         "optimism": round(r2_in - r2_cv, 3)})
            print("  %-8s %-12s in-sample %.3f  adj %.3f  CV %.3f  optimism %.3f"
                  % (season, idx, r2_in, r2_adj, r2_cv, r2_in - r2_cv), flush=True)

    out = pd.DataFrame(rows)
    f = os.path.join(RESD, "stepwise_vs_crossvalidated.csv")
    out.to_csv(f, index=False)
    print("\nwrote", f)
    return out


def figure(out):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True,
                             sharey=True)
    for ax, season, col in zip(axes, [S1, S2], [C_S1, C_S2]):
        d = out[out.Season == season].set_index("Index").reindex(INDICES)
        x = np.arange(len(d))
        ax.bar(x - 0.21, d.R2_in_sample, 0.42, color=col, label="In-sample, after AIC selection")
        ax.bar(x + 0.21, d.R2_cross_validated, 0.42, color="#A0AEC0",
               label="Cross-validated (selection inside each fold)")
        for xi, (a, b) in enumerate(zip(d.R2_in_sample, d.R2_cross_validated)):
            ax.annotate("", xy=(xi + 0.21, b), xytext=(xi - 0.21, a),
                        arrowprops=dict(arrowstyle="->", color="#9B2226", lw=1.1))
            ax.text(xi, max(a, b) + 0.03, "%.2f" % (a - b), ha="center", fontsize=7.5,
                    color="#9B2226", fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(INDICES, rotation=25, ha="right")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(season, fontsize=10, fontweight="bold")
        ax.set_ylim(-0.1, 1.0)
    axes[0].set_ylabel("$R^2$ for grain yield")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Optimism of AIC-selected multiple linear regression on "
                 "double-logistic parameters", fontsize=12, fontweight="bold")
    p = os.path.join(FIGD, "figM1_stepwise_optimism.png")
    fig.savefig(p); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    out = run()
    figure(out)
    print("\nMEAN OPTIMISM: %.3f R2 units" % out.optimism.mean())
    print(out.groupby("Season")[["R2_in_sample", "R2_cross_validated", "optimism"]]
          .mean().round(3).to_string())
