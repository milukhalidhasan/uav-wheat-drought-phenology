"""
Per-model performance tables: every one of the 12 regressors, in both seasons and
across seasons, for every predictor set and every yield target.

Writes:
  04_ML/results/model_performance.xlsx      one sheet per target x metric (all predictor sets)
  04_ML/results/model_performance_R2.csv    flat long form, every metric
  04_ML/results/best_model_summary.csv      best model per target x predictor set x scheme
  04_ML/figures/fig6_models_<target>.png    all models x all schemes, per target
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_ml_analysis as R

RESD, FIGD = R.RESD, os.path.join(R.OUTD, "figures")

MODELS = ["Dummy_mean", "OLS", "Ridge", "Lasso", "ElasticNet", "PLSR", "SVR_RBF",
          "RandomForest", "ExtraTrees", "HistGB", "GPR", "KNN"]
FSETS = ["Design_only", "Agro_predictive", "Agro_predictive+Design", "Agro_full",
         "Pheno", "Pheno+Design"]
SCHEMES = ["CV_2024-25", "Block_2024-25", "CV_2025-26", "Block_2025-26", "CV_pooled",
           "Transfer_24to25", "Transfer_25to24", "TransferZ_24to25", "TransferZ_25to24"]
NICE = {"CV_2024-25": "CV 24-25", "Block_2024-25": "Block 24-25",
        "CV_2025-26": "CV 25-26", "Block_2025-26": "Block 25-26",
        "CV_pooled": "Pooled", "Transfer_24to25": "Tr 24>25",
        "Transfer_25to24": "Tr 25>24", "TransferZ_24to25": "TrZ 24>25",
        "TransferZ_25to24": "TrZ 25>24"}

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "figure.facecolor": "white", "savefig.bbox": "tight"})


def table(res, target, fset, metric="R2_mean"):
    d = res[(res.Target == target) & (res.Feature_set == fset)]
    t = d.pivot(index="Model", columns="Scheme", values=metric)
    return t.reindex(index=MODELS, columns=SCHEMES).rename(columns=NICE)


def write_workbook(res, xlsx):
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        pd.DataFrame({
            "Item": ["Contents", "Rows", "Columns", "R2_mean", "R2_sd", "Transfer",
                     "TransferZ", "Block", "Pooled", "Dummy_mean", "OLS on Pheno",
                     "Source"],
            "Description": [
                "Every model x predictor set x validation scheme, for each yield target",
                "The 12 regressors compared",
                "The 9 validation schemes",
                "Mean R2 over repeats of pooled out-of-fold predictions",
                "SD across the 3 repeats; 0 for single-fit schemes (transfers)",
                "Train one season, predict the other, raw values",
                "Same, but X and y standardised within each season first",
                "Leave-one-replication-out, generalising to an unseen block",
                "Both seasons together with Season as a predictor",
                "Mean-only baseline; any usable model must beat this",
                "Large negative values are the expected p > n failure, kept as a control",
                "ml_results_all.csv, produced by run_ml_analysis.py"]}
        ).to_excel(xw, sheet_name="README", index=False)

        for target in R.TARGETS:
            for metric, tag in [("R2_mean", "R2"), ("RMSE_mean", "RMSE")]:
                rows = []
                for fs in FSETS:
                    t = table(res, target, fs, metric).round(3)
                    t.insert(0, "Predictor_set", fs)
                    rows.append(t.reset_index())
                sheet = ("%s_%s" % (target.replace("_t_ha", "").replace("_", ""), tag))[:31]
                pd.concat(rows, ignore_index=True).to_excel(xw, sheet_name=sheet, index=False)


def figures(res):
    for target in R.TARGETS:
        fig, axes = plt.subplots(2, 1, figsize=(11.5, 8), constrained_layout=True)
        for ax, fs in zip(axes, ["Agro_predictive", "Pheno"]):
            raw = table(res, target, fs)
            t = raw.clip(lower=-0.35)
            im = ax.imshow(t.values, cmap="RdYlGn", vmin=-0.35, vmax=0.9, aspect="auto")
            ax.set_xticks(range(t.shape[1]))
            ax.set_xticklabels(t.columns, rotation=30, ha="right")
            ax.set_yticks(range(t.shape[0]))
            ax.set_yticklabels(t.index)
            for i in range(t.shape[0]):
                for j in range(t.shape[1]):
                    v, rv = t.values[i, j], raw.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, ("%.2f" % rv) if rv > -10 else "<<0",
                                ha="center", va="center", fontsize=7.5,
                                color="black" if -0.2 < v < 0.65 else "white")
            ax.grid(False)
            ax.set_title("%s" % fs, loc="left", fontsize=10, fontweight="bold")
            fig.colorbar(im, ax=ax, label="$R^2$", shrink=0.9)
        fig.suptitle("Individual model performance: %s" % target.replace("_", " "),
                     fontsize=12, fontweight="bold")
        tag = target.replace("_t_ha", "").replace("_", "")
        fig.savefig(os.path.join(FIGD, "fig6_models_%s.png" % tag))
        plt.close(fig)


def main():
    os.makedirs(FIGD, exist_ok=True)
    res = pd.read_csv(os.path.join(RESD, "ml_results_all.csv"))

    res[["Target", "Feature_set", "Model", "Scheme", "n_features", "R2_mean", "R2_sd",
         "RMSE_mean", "RMSE_sd", "MAE_mean", "bias_mean", "best_params"]] \
        .to_csv(os.path.join(RESD, "model_performance_R2.csv"), index=False)

    xlsx = os.path.join(RESD, "model_performance.xlsx")
    write_workbook(res, xlsx)
    figures(res)

    best = (res[res.Model != "Dummy_mean"].sort_values("R2_mean", ascending=False)
            .groupby(["Target", "Feature_set", "Scheme"]).first().reset_index()
            [["Target", "Feature_set", "Scheme", "Model", "n_features",
              "R2_mean", "RMSE_mean", "bias_mean"]])
    best.to_csv(os.path.join(RESD, "best_model_summary.csv"), index=False)

    for target in R.TARGETS:
        for fs in FSETS:
            print("\n" + "=" * 112)
            print("%s  |  %s  |  R2 per model" % (target, fs))
            print(table(res, target, fs).round(3).to_string())
    print("\nwrote", xlsx)


if __name__ == "__main__":
    main()
