"""
Machine learning analysis of the two-season wheat dataset.

Two predictor domains, as planned:
  A. agronomic / physiological parameters  -> yield parameters
  B. double-logistic phenology parameters  -> yield parameters

Each is evaluated within 2024-25, within 2025-26, pooled across both, and
transferred between seasons.

Design notes that drive the choices here
----------------------------------------
* n = 96 per season, 192 pooled. Small. No deep learning; nested cross
  validation everywhere so hyper-parameter choice never sees the test fold.
* Phenology has 138 predictors for 96 samples (p > n) and is strongly
  collinear, so shrinkage and latent-variable methods (PLSR) belong in the
  comparison alongside tree ensembles.
* The trial is blocked (4 replications). Plain k-fold leaks block effects, so
  leave-one-replication-out is reported next to repeated k-fold.
* Several "agronomic" traits are yield COMPONENTS measured on the same
  harvested plants. Predicting grain yield from them is partly definitional,
  so the honest (non-component) and full trait sets are reported separately.
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from wheat_paths import (ROOT, F24, F25, OUT, CSVD, XLSX, RESD, FIGD,
                         ML_DIR, REPORT_DOCX, check_inputs)

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")   # inherited by the worker processes

OUTD = ML_DIR
SEED = 20260919
S1, S2 = "2024-25", "2025-26"

KEYCOLS = ["Season", "Season_ID", "Plot_UID", "Plot_ID", "Water_Regime", "Treatment",
           "N_Level", "N_kg_ha", "Seaweed_Level", "Seaweed_pct", "Replication"]
DESIGN_CAT = ["Water_Regime", "N_Level", "Seaweed_Level"]
TARGETS = ["Grain_Yield_t_ha", "Biological_Yield_t_ha", "Harvest_Index"]

# traits that are yield components / definitionally tied to a target
COMPONENTS = ["Tiller_No", "Spike_Length_cm", "Total_Grain_per_Plant", "TSW_g",
              "Fresh_Biomass_t_ha"]
# never allowed as a predictor of a given target (circularity)
FORBIDDEN = {
    "Grain_Yield_t_ha": ["Biological_Yield_t_ha", "Harvest_Index"],
    "Biological_Yield_t_ha": ["Grain_Yield_t_ha", "Harvest_Index"],
    "Harvest_Index": ["Grain_Yield_t_ha", "Biological_Yield_t_ha"],
}


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def model_zoo(n_features):
    ncomp = [c for c in (1, 2, 3, 4, 6, 8, 10, 15) if c <= min(n_features, 60)] or [1]
    alphas = np.logspace(-3, 3, 13)
    return {
        "Dummy_mean": (DummyRegressor(strategy="mean"), {}),
        "OLS": (LinearRegression(), {}),
        "Ridge": (Ridge(random_state=None), {"model__alpha": alphas}),
        "Lasso": (Lasso(max_iter=20000), {"model__alpha": np.logspace(-4, 1, 12)}),
        "ElasticNet": (ElasticNet(max_iter=20000),
                       {"model__alpha": np.logspace(-4, 1, 6),
                        "model__l1_ratio": [0.2, 0.5, 0.8]}),
        "PLSR": (PLSRegression(scale=False), {"model__n_components": ncomp}),
        "SVR_RBF": (SVR(kernel="rbf"),
                    {"model__C": [1, 10, 100, 1000], "model__gamma": ["scale", 0.01, 0.1],
                     "model__epsilon": [0.05, 0.1]}),
        "RandomForest": (RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1),
                         {"model__max_features": [0.3, 0.7],
                          "model__min_samples_leaf": [1, 3]}),
        "ExtraTrees": (ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=1),
                       {"model__max_features": [0.3, 0.7],
                        "model__min_samples_leaf": [1, 3]}),
        "HistGB": (HistGradientBoostingRegressor(random_state=SEED),
                   {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [7, 15]}),
        "GPR": (GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0) * RBF(length_scale=np.sqrt(max(n_features, 1)))
                           + WhiteKernel(0.1),
                    normalize_y=True, alpha=1e-8, random_state=SEED), {}),
        "KNN": (KNeighborsRegressor(), {"model__n_neighbors": [3, 5, 7, 10],
                                        "model__weights": ["uniform", "distance"]}),
    }


def make_pipe(estimator, num_cols, cat_cols):
    # sparse_output=False matters: PLSR and HistGradientBoosting reject sparse X
    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                               sparse_output=False), cat_cols)],
        remainder="drop")
    return Pipeline([("pre", pre), ("model", estimator)])


def fit_predict(estimator, grid, num_cols, cat_cols, Xtr, ytr, Xte, inner=4):
    pipe = make_pipe(clone(estimator), num_cols, cat_cols)
    if grid:
        n_inner = min(inner, max(2, len(ytr) // 8))
        gs = GridSearchCV(pipe, grid, cv=n_inner, scoring="neg_mean_squared_error", n_jobs=1)
        gs.fit(Xtr, ytr)
        best, params = gs.best_estimator_, gs.best_params_
    else:
        pipe.fit(Xtr, ytr)
        best, params = pipe, {}
    return np.asarray(best.predict(Xte)).ravel(), params


def metrics(y, yhat):
    return {"R2": r2_score(y, yhat),
            "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
            "MAE": float(mean_absolute_error(y, yhat)),
            "bias": float(np.mean(yhat - y))}


# ---------------------------------------------------------------------------
# evaluation schemes
# ---------------------------------------------------------------------------
def eval_cv(est, grid, df, feats, cat, target, n_splits=5, n_repeats=3, groups=None):
    """Repeated k-fold, or leave-one-group-out when groups is given.
    Metrics are computed on pooled out-of-fold predictions within each repeat."""
    num = [c for c in feats if c not in cat]
    X, y = df[feats + cat], df[target].to_numpy(float)
    if groups is not None:
        splitter = GroupKFold(n_splits=df[groups].nunique())
        splits = list(splitter.split(X, y, groups=df[groups]))
        rep_of = [0] * len(splits)
    else:
        splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
        splits = list(splitter.split(X, y))
        rep_of = [i // n_splits for i in range(len(splits))]

    n_rep = max(rep_of) + 1
    oof = np.full((n_rep, len(y)), np.nan)
    params0 = None
    for (tr, te), rep in zip(splits, rep_of):
        pred, pars = fit_predict(est, grid, num, cat, X.iloc[tr], y[tr], X.iloc[te])
        oof[rep, te] = pred
        if params0 is None:
            params0 = pars
    per_rep = [metrics(y, oof[r]) for r in range(n_rep)]
    out = {k + "_mean": float(np.mean([m[k] for m in per_rep])) for k in per_rep[0]}
    out.update({k + "_sd": float(np.std([m[k] for m in per_rep])) for k in per_rep[0]})
    out["oof"] = oof.mean(axis=0)
    out["best_params"] = params0
    return out


def eval_transfer(est, grid, tr_df, te_df, feats, cat, target, zscore=False):
    """Train on one season, test on the other. With zscore=True both X and y
    are standardised inside each season first, so the metric asks whether the
    relationship transfers rather than whether the absolute level does."""
    num = [c for c in feats if c not in cat]
    tr, te = tr_df.copy(), te_df.copy()
    if zscore:
        for d in (tr, te):
            for c in num + [target]:
                s = d[c].std(ddof=0)
                d[c] = (d[c] - d[c].mean()) / (s if s > 0 else 1.0)
    ytr, yte = tr[target].to_numpy(float), te[target].to_numpy(float)
    pred, pars = fit_predict(est, grid, num, cat, tr[feats + cat], ytr, te[feats + cat])
    out = {k + "_mean": v for k, v in metrics(yte, pred).items()}
    out.update({k + "_sd": 0.0 for k in ["R2", "RMSE", "MAE", "bias"]})
    out["oof"] = pred
    out["best_params"] = pars
    return out


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load():
    agro = pd.read_csv(os.path.join(CSVD, "ML_Agronomic_CrossSeason.csv"))
    phen = pd.read_csv(os.path.join(CSVD, "ML_Phenology_CrossSeason.csv"))
    pheno_feats = [c for c in phen.columns
                   if c not in KEYCOLS + TARGETS and not c.endswith("_R2")]
    agro_all = [c for c in agro.columns if c not in KEYCOLS + TARGETS]
    agro_pred = [c for c in agro_all if c not in COMPONENTS]
    merged = agro.merge(phen[["Plot_UID"] + pheno_feats], on="Plot_UID", how="inner")
    assert len(merged) == 192, len(merged)
    return merged, agro_pred, agro_all, pheno_feats


def feature_sets(agro_pred, agro_all, pheno_feats):
    return {
        "Design_only":        ([], DESIGN_CAT),
        "Agro_predictive":    (agro_pred, []),
        "Agro_predictive+Design": (agro_pred, DESIGN_CAT),
        "Agro_full":          (agro_all, []),
        "Pheno":              (pheno_feats, []),
        "Pheno+Design":       (pheno_feats, DESIGN_CAT),
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def one_job(df, name, feats, cat, target, model_name, est, grid, scheme):
    feats = [f for f in feats if f not in FORBIDDEN[target]]
    d1, d2 = df[df.Season == S1], df[df.Season == S2]
    try:
        if scheme == "CV_2024-25":
            r = eval_cv(est, grid, d1, feats, cat, target)
        elif scheme == "CV_2025-26":
            r = eval_cv(est, grid, d2, feats, cat, target)
        elif scheme == "Block_2024-25":
            r = eval_cv(est, grid, d1, feats, cat, target, groups="Replication")
        elif scheme == "Block_2025-26":
            r = eval_cv(est, grid, d2, feats, cat, target, groups="Replication")
        elif scheme == "CV_pooled":
            r = eval_cv(est, grid, df, feats, cat + ["Season"], target)
        elif scheme == "Transfer_24to25":
            r = eval_transfer(est, grid, d1, d2, feats, cat, target)
        elif scheme == "Transfer_25to24":
            r = eval_transfer(est, grid, d2, d1, feats, cat, target)
        elif scheme == "TransferZ_24to25":
            r = eval_transfer(est, grid, d1, d2, feats, cat, target, zscore=True)
        elif scheme == "TransferZ_25to24":
            r = eval_transfer(est, grid, d2, d1, feats, cat, target, zscore=True)
        else:
            raise ValueError(scheme)
    except Exception as exc:                                   # keep the sweep alive
        return {"Target": target, "Feature_set": name, "Model": model_name,
                "Scheme": scheme, "error": repr(exc)[:200]}
    row = {"Target": target, "Feature_set": name, "Model": model_name, "Scheme": scheme,
           "n_features": len(feats) + len(cat)}
    row.update({k: v for k, v in r.items() if k not in ("oof", "best_params")})
    row["best_params"] = json.dumps({k.replace("model__", ""): (str(v) if not isinstance(v, (int, float)) else v)
                                     for k, v in (r["best_params"] or {}).items()})
    return row


SCHEMES = ["CV_2024-25", "CV_2025-26", "Block_2024-25", "Block_2025-26", "CV_pooled",
           "Transfer_24to25", "Transfer_25to24", "TransferZ_24to25", "TransferZ_25to24"]


def main():
    os.makedirs(RESD, exist_ok=True)
    df, agro_pred, agro_all, pheno_feats = load()
    fsets = feature_sets(agro_pred, agro_all, pheno_feats)

    print("data: %d rows, %d per season" % (len(df), len(df) // 2))
    for k, (f, c) in fsets.items():
        print("  feature set %-24s %3d numeric + %d categorical" % (k, len(f), len(c)))

    jobs = []
    for target in TARGETS:
        for name, (feats, cat) in fsets.items():
            zoo = model_zoo(len(feats) + len(cat))
            for mname, (est, grid) in zoo.items():
                for scheme in SCHEMES:
                    jobs.append((df, name, feats, cat, target, mname, est, grid, scheme))
    print("\n%d jobs -> running on all cores" % len(jobs))

    rows = Parallel(n_jobs=-1, verbose=5, batch_size=8)(delayed(one_job)(*j) for j in jobs)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESD, "ml_results_all.csv"), index=False)

    bad = res[res.get("error").notna()] if "error" in res else res.iloc[:0]
    if len(bad):
        print("\n%d jobs failed:" % len(bad))
        print(bad.groupby(["Model", "Scheme"]).size().to_string())

    ok = res[res.R2_mean.notna()] if "R2_mean" in res else res
    ok.to_csv(os.path.join(RESD, "ml_results_clean.csv"), index=False)
    print("\nwrote", os.path.join(RESD, "ml_results_all.csv"), "(%d rows)" % len(res))
    return res


if __name__ == "__main__":
    main()
