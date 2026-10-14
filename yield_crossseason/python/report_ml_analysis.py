"""
Summarise the ML sweep: headline tables, figures, and out-of-fold predictions
plus permutation importance for the winning model of each domain.
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold

import run_ml_analysis as R

warnings.filterwarnings("ignore")

RESD = R.RESD
FIGD = os.path.join(R.OUTD, "figures")
S1, S2 = R.S1, R.S2
PRIMARY = "Grain_Yield_t_ha"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "figure.facecolor": "white", "savefig.bbox": "tight",
})
C_S1, C_S2 = "#2B6CB0", "#C05621"


# ---------------------------------------------------------------------------
def load_results():
    res = pd.read_csv(os.path.join(RESD, "ml_results_all.csv"))
    return res[res.R2_mean.notna()].copy()


def fig_model_comparison(res):
    """Model x feature-set R2 for grain yield, within each season."""
    d = res[(res.Target == PRIMARY) & (res.Scheme.isin(["CV_2024-25", "CV_2025-26"]))]
    fsets = ["Design_only", "Agro_predictive", "Agro_predictive+Design", "Agro_full",
             "Pheno", "Pheno+Design"]
    fsets = [f for f in fsets if f in set(d.Feature_set)]
    models = [m for m in ["Dummy_mean", "OLS", "Ridge", "Lasso", "ElasticNet", "PLSR",
                          "SVR_RBF", "RandomForest", "ExtraTrees", "HistGB", "GPR", "KNN"]
              if m in set(d.Model)]

    fig, axes = plt.subplots(len(fsets), 1, figsize=(9, 2.1 * len(fsets)), sharex=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(models))
    for ax, fs in zip(axes, fsets):
        for off, (sch, col, lab) in enumerate([("CV_2024-25", C_S1, S1), ("CV_2025-26", C_S2, S2)]):
            sub = d[(d.Feature_set == fs) & (d.Scheme == sch)].set_index("Model").reindex(models)
            ax.bar(x + (off - 0.5) * 0.4, sub.R2_mean.clip(lower=-0.25), 0.4,
                   yerr=sub.R2_sd, color=col, label=lab, capsize=2, error_kw={"lw": 0.7})
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("$R^2$")
        ax.set_title(fs, loc="left", fontsize=9, fontweight="bold")
        ax.set_ylim(-0.3, 1.0)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(models, rotation=35, ha="right")
    fig.suptitle("Grain yield: nested-CV $R^2$ by model and predictor set, within season",
                 y=0.995, fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(FIGD, "fig1_model_comparison.png"))
    plt.close(fig)


def fig_scheme_comparison(res):
    """Best model per feature set under each validation scheme."""
    d = res[res.Target == PRIMARY]
    schemes = ["CV_2024-25", "Block_2024-25", "CV_2025-26", "Block_2025-26", "CV_pooled",
               "Transfer_24to25", "Transfer_25to24", "TransferZ_24to25", "TransferZ_25to24"]
    schemes = [s for s in schemes if s in set(d.Scheme)]
    fsets = [f for f in ["Design_only", "Agro_predictive", "Agro_predictive+Design",
                         "Agro_full", "Pheno", "Pheno+Design"] if f in set(d.Feature_set)]
    best = (d[d.Model != "Dummy_mean"]
            .sort_values("R2_mean", ascending=False)
            .groupby(["Feature_set", "Scheme"]).first().reset_index())
    mat = best.pivot(index="Feature_set", columns="Scheme", values="R2_mean").reindex(
        index=fsets, columns=schemes)

    fig, ax = plt.subplots(figsize=(10, 3.4))
    norm = TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.9)
    im = ax.imshow(mat.values, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels(schemes, rotation=35, ha="right")
    ax.set_yticks(range(len(fsets)))
    ax.set_yticklabels(fsets)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=8,
                        color="black" if -0.25 < v < 0.65 else "white")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="$R^2$ of best model", shrink=0.85)
    ax.set_title("Grain yield: best achievable $R^2$ by predictor set and validation scheme",
                 fontsize=11, fontweight="bold", loc="left")
    fig.savefig(os.path.join(FIGD, "fig2_validation_schemes.png"))
    plt.close(fig)


def fig_targets(res):
    """All three yield targets, best model, within season."""
    d = res[(res.Model != "Dummy_mean") & (res.Scheme.isin(["CV_2024-25", "CV_2025-26"]))]
    best = d.sort_values("R2_mean", ascending=False) \
            .groupby(["Target", "Feature_set", "Scheme"]).first().reset_index()
    fsets = [f for f in ["Design_only", "Agro_predictive", "Agro_full", "Pheno"]
             if f in set(best.Feature_set)]
    fig, axes = plt.subplots(1, len(R.TARGETS), figsize=(11, 3.6), sharey=True,
                             constrained_layout=True)
    x = np.arange(len(fsets))
    for ax, tgt in zip(np.atleast_1d(axes), R.TARGETS):
        for off, (sch, col, lab) in enumerate([("CV_2024-25", C_S1, S1), ("CV_2025-26", C_S2, S2)]):
            sub = best[(best.Target == tgt) & (best.Scheme == sch)] \
                    .set_index("Feature_set").reindex(fsets)
            ax.bar(x + (off - 0.5) * 0.4, sub.R2_mean.clip(lower=-0.15), 0.4,
                   color=col, label=lab)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(tgt.replace("_", " "), fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(fsets, rotation=35, ha="right")
        ax.set_ylim(-0.2, 1.0)
    np.atleast_1d(axes)[0].set_ylabel("$R^2$ of best model")
    np.atleast_1d(axes)[0].legend(frameon=False)
    fig.suptitle("Every yield target, best model per predictor set", fontsize=11,
                 fontweight="bold")
    fig.savefig(os.path.join(FIGD, "fig3_targets.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
def refit_oof(df, feats, cat, target, model_name, season):
    """Out-of-fold predictions + permutation importance for one configuration."""
    d = df[df.Season == season].reset_index(drop=True)
    feats = [f for f in feats if f not in R.FORBIDDEN[target]]
    num = [c for c in feats if c not in cat]
    zoo = R.model_zoo(len(feats) + len(cat))
    est, grid = zoo[model_name]
    X, y = d[feats + cat], d[target].to_numpy(float)

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=R.SEED)
    oof = np.full((3, len(y)), np.nan)
    imps = []
    for i, (tr, te) in enumerate(cv.split(X)):
        pipe = R.make_pipe(est, num, cat)
        if grid:
            from sklearn.model_selection import GridSearchCV
            gs = GridSearchCV(pipe, grid, cv=4, scoring="neg_mean_squared_error", n_jobs=-1)
            gs.fit(X.iloc[tr], y[tr])
            fitted = gs.best_estimator_
        else:
            pipe.fit(X.iloc[tr], y[tr])
            fitted = pipe
        oof[i // 5, te] = np.asarray(fitted.predict(X.iloc[te])).ravel()
        pi = permutation_importance(fitted, X.iloc[te], y[te], n_repeats=10,
                                    random_state=R.SEED, scoring="r2", n_jobs=-1)
        imps.append(pd.Series(pi.importances_mean, index=feats + cat))
    return d, oof.mean(axis=0), pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)


def fig_obs_pred(panels):
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.4), constrained_layout=True)
    for ax, (title, d, pred, r2) in zip(axes.ravel(), panels):
        col = C_S1 if S1 in title else C_S2
        for wr, mk in [("D", "o"), ("I", "^")]:
            m = d.Water_Regime == wr
            ax.scatter(d[PRIMARY][m], pred[m.to_numpy()], s=26, marker=mk, alpha=0.75,
                       edgecolor="white", linewidth=0.5, color=col,
                       label="Drought" if wr == "D" else "Irrigated")
        lo = min(d[PRIMARY].min(), np.nanmin(pred)) - 0.3
        hi = max(d[PRIMARY].max(), np.nanmax(pred)) + 0.3
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Observed grain yield (t/ha)")
        ax.set_ylabel("Out-of-fold prediction (t/ha)")
        ax.set_title("%s\n$R^2$ = %.2f" % (title, r2), fontsize=9, fontweight="bold")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Observed vs out-of-fold predicted grain yield", fontsize=11,
                 fontweight="bold")
    fig.savefig(os.path.join(FIGD, "fig4_obs_vs_pred.png"))
    plt.close(fig)


def fig_importance(imps):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), constrained_layout=True)
    for ax, (title, s) in zip(axes.ravel(), imps):
        top = s.head(15)[::-1]
        col = C_S1 if S1 in title else C_S2
        ax.barh(range(len(top)), top.values, color=col)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([t.replace("_", " ") for t in top.index], fontsize=7.5)
        ax.set_xlabel("drop in $R^2$ when permuted")
        ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
        ax.axvline(0, color="black", lw=0.8)
    fig.suptitle("Permutation importance, held-out folds (top 15)", fontsize=11,
                 fontweight="bold")
    fig.savefig(os.path.join(FIGD, "fig5_importance.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIGD, exist_ok=True)
    res = load_results()
    df, ap, aa, pf = R.load()
    fsets = R.feature_sets(ap, aa, pf)

    print("figures 1-3 ...")
    fig_model_comparison(res)
    fig_scheme_comparison(res)
    fig_targets(res)

    # headline tables
    d = res[res.Target == PRIMARY]
    best = (d[d.Model != "Dummy_mean"].sort_values("R2_mean", ascending=False)
            .groupby(["Feature_set", "Scheme"]).first().reset_index())
    best[["Feature_set", "Scheme", "Model", "n_features", "R2_mean", "R2_sd",
          "RMSE_mean", "RMSE_sd", "MAE_mean", "bias_mean", "best_params"]] \
        .to_csv(os.path.join(RESD, "best_per_featureset_scheme.csv"), index=False)

    top = (res[res.Model != "Dummy_mean"].sort_values("R2_mean", ascending=False)
           .groupby(["Target", "Scheme"]).head(3))
    top.to_csv(os.path.join(RESD, "top3_per_target_scheme.csv"), index=False)

    # winning model per domain, within each season
    panels, imps, oof_rows = [], [], []
    for dom, fs in [("Agronomic", "Agro_predictive"), ("Phenology", "Pheno")]:
        for season, sch in [(S1, "CV_2024-25"), (S2, "CV_2025-26")]:
            sel = d[(d.Feature_set == fs) & (d.Scheme == sch) & (d.Model != "Dummy_mean")]
            mname = sel.sort_values("R2_mean", ascending=False).iloc[0].Model
            feats, cat = fsets[fs]
            print("refit %-10s %-8s -> %s" % (dom, season, mname))
            dd, pred, imp = refit_oof(df, feats, cat, PRIMARY, mname, season)
            from sklearn.metrics import r2_score
            panels.append(("%s features, %s (%s)" % (dom, season, mname), dd, pred,
                           r2_score(dd[PRIMARY], pred)))
            imps.append(("%s features, %s (%s)" % (dom, season, mname), imp))
            oof_rows.append(dd[["Season", "Plot_UID", "Water_Regime", "N_kg_ha",
                                "Seaweed_pct", "Replication", PRIMARY]]
                            .assign(Domain=dom, Model=mname, Predicted=pred))
            imp.rename("importance").to_frame().to_csv(
                os.path.join(RESD, "importance_%s_%s.csv" % (dom, season.replace("-", ""))))

    print("figures 4-5 ...")
    fig_obs_pred(panels)
    fig_importance(imps)
    pd.concat(oof_rows).to_csv(os.path.join(RESD, "oof_predictions.csv"), index=False)

    print("\nBEST PER PREDICTOR SET (grain yield)")
    show = best.pivot(index="Feature_set", columns="Scheme", values="R2_mean")
    print(show.round(3).to_string())
    print("\nfigures ->", FIGD)


if __name__ == "__main__":
    main()
