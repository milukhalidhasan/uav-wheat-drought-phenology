"""
Extended ML sweep: every predictor block, including the ones the first sweep left
out (spectral time series, time-resolved ground data, the 2024-25-only extras).

Same nested-CV protocol as run_ml_analysis.py. Tree grids are trimmed for the very
wide blocks (>200 predictors) so the sweep stays tractable; everything else is
identical, and the narrow blocks reproduce the first sweep.
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

import extended_features as E

from wheat_paths import (ROOT, F24, F25, OUT, CSVD, XLSX, RESD, FIGD,
                         ML_DIR, REPORT_DOCX, check_inputs)

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

SEED = 20260919
S1, S2 = E.S1, E.S2
WIDE = 200                                   # predictors above which grids are trimmed


def model_zoo(n_features):
    wide = n_features > WIDE
    ncomp = [c for c in (1, 2, 3, 4, 6, 8, 10, 15) if c <= min(n_features, 60)] or [1]
    ntree = 200 if wide else 300
    mfeat = [0.1, 0.3] if wide else [0.3, 0.7]
    return {
        "Dummy_mean": (DummyRegressor(strategy="mean"), {}),
        "OLS": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"model__alpha": np.logspace(-3, 3, 13)}),
        "Lasso": (Lasso(max_iter=20000), {"model__alpha": np.logspace(-4, 1, 12)}),
        "ElasticNet": (ElasticNet(max_iter=20000),
                       {"model__alpha": np.logspace(-4, 1, 6),
                        "model__l1_ratio": [0.2, 0.5, 0.8]}),
        "PLSR": (PLSRegression(scale=False), {"model__n_components": ncomp}),
        "SVR_RBF": (SVR(kernel="rbf"),
                    {"model__C": [1, 10, 100, 1000], "model__gamma": ["scale", 0.01, 0.1],
                     "model__epsilon": [0.05, 0.1]}),
        "RandomForest": (RandomForestRegressor(n_estimators=ntree, random_state=SEED, n_jobs=1),
                         {"model__max_features": mfeat, "model__min_samples_leaf": [1, 3]}),
        "ExtraTrees": (ExtraTreesRegressor(n_estimators=ntree, random_state=SEED, n_jobs=1),
                       {"model__max_features": mfeat, "model__min_samples_leaf": [1, 3]}),
        "HistGB": (HistGradientBoostingRegressor(random_state=SEED),
                   {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [7, 15]}),
        "GPR": (GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0) * RBF(length_scale=np.sqrt(max(n_features, 1)))
                           + WhiteKernel(0.1),
                    normalize_y=True, alpha=1e-8, random_state=SEED), {}),
        "KNN": (KNeighborsRegressor(), {"model__n_neighbors": [3, 5, 7, 10],
                                        "model__weights": ["uniform", "distance"]}),
    }


def make_pipe(est, num_cols, cat_cols):
    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                               sparse_output=False), cat_cols)], remainder="drop")
    return Pipeline([("pre", pre), ("model", est)])


def fit_predict(est, grid, num, cat, Xtr, ytr, Xte, inner=4):
    pipe = make_pipe(clone(est), num, cat)
    if grid:
        n_inner = min(inner, max(2, len(ytr) // 8))
        gs = GridSearchCV(pipe, grid, cv=n_inner, scoring="neg_mean_squared_error", n_jobs=1)
        gs.fit(Xtr, ytr)
        return np.asarray(gs.best_estimator_.predict(Xte)).ravel(), gs.best_params_
    pipe.fit(Xtr, ytr)
    return np.asarray(pipe.predict(Xte)).ravel(), {}


def metrics(y, yhat):
    return {"R2": r2_score(y, yhat), "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
            "MAE": float(mean_absolute_error(y, yhat)), "bias": float(np.mean(yhat - y))}


def eval_cv(est, grid, d, feats, cat, target, groups=None, n_splits=5, n_repeats=3):
    num = [c for c in feats if c not in cat]
    X, y = d[feats + cat], d[target].to_numpy(float)
    if groups is not None:
        splits = list(GroupKFold(n_splits=d[groups].nunique()).split(X, y, groups=d[groups]))
        rep_of = [0] * len(splits)
    else:
        splits = list(RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                    random_state=SEED).split(X, y))
        rep_of = [i // n_splits for i in range(len(splits))]
    oof = np.full((max(rep_of) + 1, len(y)), np.nan)
    p0 = None
    for (tr, te), rep in zip(splits, rep_of):
        pred, pars = fit_predict(est, grid, num, cat, X.iloc[tr], y[tr], X.iloc[te])
        oof[rep, te] = pred
        p0 = p0 if p0 is not None else pars
    per = [metrics(y, oof[r]) for r in range(oof.shape[0])]
    out = {k + "_mean": float(np.mean([m[k] for m in per])) for k in per[0]}
    out.update({k + "_sd": float(np.std([m[k] for m in per])) for k in per[0]})
    out["best_params"] = p0
    return out


def eval_transfer(est, grid, tr_df, te_df, feats, cat, target, zscore=False):
    num = [c for c in feats if c not in cat]
    tr, te = tr_df.copy(), te_df.copy()
    if zscore:
        for d in (tr, te):
            for c in num + [target]:
                s = d[c].std(ddof=0)
                d[c] = (d[c] - d[c].mean()) / (s if s > 0 else 1.0)
    pred, pars = fit_predict(est, grid, num, cat, tr[feats + cat],
                             tr[target].to_numpy(float), te[feats + cat])
    out = {k + "_mean": v for k, v in metrics(te[target].to_numpy(float), pred).items()}
    out.update({k + "_sd": 0.0 for k in ["R2", "RMSE", "MAE", "bias"]})
    out["best_params"] = pars
    return out


SCHEMES_BOTH = ["CV_2024-25", "Block_2024-25", "CV_2025-26", "Block_2025-26", "CV_pooled",
                "Transfer_24to25", "Transfer_25to24", "TransferZ_24to25", "TransferZ_25to24"]
SCHEMES_WITHIN = ["CV_2024-25", "Block_2024-25", "CV_2025-26", "Block_2025-26"]
SCHEMES_S1 = ["CV_2024-25", "Block_2024-25"]


def one_job(df, name, feats, cat, target, mname, est, grid, scheme):
    d1, d2 = df[df.Season == S1], df[df.Season == S2]
    f1 = [f for f in E.cols_present(df, feats, S1) if f not in E.FORBIDDEN[target]]
    f2 = [f for f in E.cols_present(df, feats, S2) if f not in E.FORBIDDEN[target]]
    fb = [f for f in f1 if f in f2]
    try:
        if scheme == "CV_2024-25":
            r = eval_cv(est, grid, d1, f1, cat, target)
        elif scheme == "CV_2025-26":
            r = eval_cv(est, grid, d2, f2, cat, target)
        elif scheme == "Block_2024-25":
            r = eval_cv(est, grid, d1, f1, cat, target, groups="Replication")
        elif scheme == "Block_2025-26":
            r = eval_cv(est, grid, d2, f2, cat, target, groups="Replication")
        elif scheme == "CV_pooled":
            r = eval_cv(est, grid, df, fb, cat + ["Season"], target)
        elif scheme == "Transfer_24to25":
            r = eval_transfer(est, grid, d1, d2, fb, cat, target)
        elif scheme == "Transfer_25to24":
            r = eval_transfer(est, grid, d2, d1, fb, cat, target)
        elif scheme == "TransferZ_24to25":
            r = eval_transfer(est, grid, d1, d2, fb, cat, target, zscore=True)
        elif scheme == "TransferZ_25to24":
            r = eval_transfer(est, grid, d2, d1, fb, cat, target, zscore=True)
        else:
            raise ValueError(scheme)
    except Exception as exc:
        return {"Target": target, "Feature_set": name, "Model": mname, "Scheme": scheme,
                "error": repr(exc)[:200]}
    nf = {"CV_2024-25": len(f1), "Block_2024-25": len(f1), "CV_2025-26": len(f2),
          "Block_2025-26": len(f2)}.get(scheme, len(fb)) + len(cat)
    row = {"Target": target, "Feature_set": name, "Model": mname, "Scheme": scheme,
           "n_features": nf}
    row.update({k: v for k, v in r.items() if k != "best_params"})
    row["best_params"] = json.dumps({k.replace("model__", ""): str(v)
                                     for k, v in (r["best_params"] or {}).items()})
    return row


def main():
    os.makedirs(RESD, exist_ok=True)
    df, blocks = E.build()
    fsets = E.feature_sets(df, blocks)

    jobs = []
    for target in E.TARGETS:
        for name, (feats, cat, scope) in fsets.items():
            schemes = {"both": SCHEMES_BOTH, "within": SCHEMES_WITHIN,
                       "s1": SCHEMES_S1}[scope]
            zoo = model_zoo(len(feats) + len(cat))
            for mname, (est, grid) in zoo.items():
                for scheme in schemes:
                    jobs.append((df, name, feats, cat, target, mname, est, grid, scheme))
    print("%d jobs" % len(jobs), flush=True)

    rows = Parallel(n_jobs=-1, verbose=5, batch_size=4)(delayed(one_job)(*j) for j in jobs)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESD, "ml_results_extended.csv"), index=False)
    if "error" in res:
        bad = res[res.error.notna()]
        print("failed:", len(bad))
        if len(bad):
            print(bad.groupby(["Feature_set", "Model"]).size().to_string())
    print("wrote", os.path.join(RESD, "ml_results_extended.csv"), len(res))


if __name__ == "__main__":
    main()
