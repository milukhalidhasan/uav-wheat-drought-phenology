"""Analysis A: can UAV canopy data tell drought plots from irrigated plots, within a season and
from one season to the next?

Feature sets: fitted double-logistic parameters, raw interpolated trajectories (6 VIs, or all
bands + VIs) and a single date at 80 DAS. Five classifiers. Validation schemes:

  LORO   within season, leave-one-replication-out (12 D + 12 I plots held out per fold)
  Tr     between seasons, raw features (train on one season, test on the other)
  TrZ    between seasons, features z-scored within each season using all plots of that season
         (label-free, but assumes the new season contains stressed and unstressed plots in
         similar proportion)
  TrRef  between seasons, features scaled against well-watered reference plots: the training
         season by all its irrigated plots, the test season by the 12 irrigated plots of ONE
         replication (a small reference block); scored on the remaining 84 plots and averaged
         over the four possible reference blocks

Hyper-parameters are tuned by an inner group-wise CV on the training data only.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import (COMMON_DAS, RESULTS, SEASONS, interp_cols, label, load_dl,
                    load_feature_sets, load_interp, zscore_within)

warnings.filterwarnings("ignore")
RNG = 0


def models():
    return {
        "LDA (shrinkage)": (make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")), None),
        "Logistic (L2)": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                          {"logisticregression__C": [0.01, 0.1, 1, 10]}),
        "SVM (RBF)": (make_pipeline(StandardScaler(), SVC()),
                      {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.01]}),
        "Random forest": (RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=RNG), None),
        "Gradient boosting": (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=RNG), None),
    }


def fit(est, grid, X, y, groups):
    if grid is None:
        return clone(est).fit(X, y)
    n = min(4, len(np.unique(groups)))
    gs = GridSearchCV(clone(est), grid, cv=GroupKFold(n_splits=n), scoring="balanced_accuracy")
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_


def score(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def metrics(y, pred, s):
    k = int((pred == y).sum())
    ci = binomtest(k, len(y)).proportion_ci(method="wilson")
    return {
        "balanced_acc": balanced_accuracy_score(y, pred),
        "auc": roc_auc_score(y, s),
        "acc": accuracy_score(y, pred),
        "acc_ci_low": ci.low, "acc_ci_high": ci.high,
        "p_vs_chance": binomtest(k, len(y), 0.5, alternative="greater").pvalue,
        "n_test": len(y),
    }


def within_season(df, cols, est, grid):
    rows = []
    for season in SEASONS:
        d = df[df.Season == season].reset_index(drop=True)
        X, y, g = d[cols].to_numpy(), label(d), d["Replication"].to_numpy()
        pred, s = np.zeros(len(y), int), np.zeros(len(y))
        for rep in np.unique(g):
            tr, te = g != rep, g == rep
            m = fit(est, grid, X[tr], y[tr], g[tr])
            pred[te], s[te] = m.predict(X[te]), score(m, X[te])
        rows.append({"scheme": "LORO", "direction": season, **metrics(y, pred, s)})
    return rows


def between_seasons(df, cols, est, grid, scheme):
    rows = []
    for a, b in [(SEASONS[0], SEASONS[1]), (SEASONS[1], SEASONS[0])]:
        if scheme == "TrRef":
            # training season scaled by all of its irrigated plots
            tr_d = df[df.Season == a].reset_index(drop=True)
            tr_d = zscore_within(tr_d, cols, ref_mask=tr_d["Water_Regime"] == "I")
            m = fit(est, grid, tr_d[cols].to_numpy(), label(tr_d), tr_d["Replication"].to_numpy())
            per_ref = []
            te_all = df[df.Season == b].reset_index(drop=True)
            for ref_rep in sorted(te_all["Replication"].unique()):
                ref = (te_all["Water_Regime"] == "I") & (te_all["Replication"] == ref_rep)
                te_d = zscore_within(te_all, cols, ref_mask=ref)
                keep = ~ref.to_numpy()
                Xte, yte = te_d[cols].to_numpy()[keep], label(te_d)[keep]
                per_ref.append(metrics(yte, m.predict(Xte), score(m, Xte)))
            avg = pd.DataFrame(per_ref).mean(numeric_only=True).to_dict()
            avg["n_test"] = int(per_ref[0]["n_test"])
            rows.append({"scheme": scheme, "direction": f"{a} -> {b}", **avg})
            continue
        d = zscore_within(df, cols) if scheme == "TrZ" else df
        tr_d, te_d = d[d.Season == a], d[d.Season == b]
        m = fit(est, grid, tr_d[cols].to_numpy(), label(tr_d), tr_d["Replication"].to_numpy())
        Xte, yte = te_d[cols].to_numpy(), label(te_d)
        rows.append({"scheme": scheme, "direction": f"{a} -> {b}", **metrics(yte, m.predict(Xte), score(m, Xte))})
    return rows


def run_main():
    out = []
    for fs_name, (df, cols) in load_feature_sets().items():
        df = df.dropna(subset=cols).reset_index(drop=True)
        for m_name, (est, grid) in models().items():
            rows = within_season(df, cols, est, grid)
            for scheme in ["Tr", "TrZ", "TrRef"]:
                rows += between_seasons(df, cols, est, grid, scheme)
            for r in rows:
                out.append({"feature_set": fs_name, "n_features": len(cols), "model": m_name, **r})
            print(f"  done: {fs_name} | {m_name}")
    res = pd.DataFrame(out)
    res.to_csv(RESULTS / "A1_classifier_all_configurations.csv", index=False)

    # best model per feature set x scheme x direction (selected on the score itself, so this is
    # an optimistic summary; the mean over models is reported alongside)
    best = (res.sort_values("balanced_acc", ascending=False)
               .groupby(["feature_set", "scheme", "direction"], as_index=False).first())
    mean = (res.groupby(["feature_set", "scheme", "direction"], as_index=False)
               [["balanced_acc", "auc"]].mean()
               .rename(columns={"balanced_acc": "mean_balanced_acc_over_models", "auc": "mean_auc_over_models"}))
    summary = best.merge(mean, on=["feature_set", "scheme", "direction"])
    summary.to_csv(RESULTS / "A2_classifier_summary_best_and_mean.csv", index=False)
    return res, summary


def run_per_das():
    """How early in the season can drought be told apart? One date at a time."""
    sp, sp_cols = load_interp()
    est_grid = models()
    rows = []
    for das in COMMON_DAS:
        cols = interp_cols(sp_cols, das=[das])
        for m_name in ["LDA (shrinkage)", "Logistic (L2)"]:
            est, grid = est_grid[m_name]
            for r in within_season(sp, cols, est, grid) + between_seasons(sp, cols, est, grid, "Tr") \
                    + between_seasons(sp, cols, est, grid, "TrZ"):
                rows.append({"DAS": das, "model": m_name, **r})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "A3_detection_by_date.csv", index=False)
    return res


def run_importance():
    """Which double-logistic parameters carry the drought signal under standardised transfer?"""
    dl, cols = load_dl()
    d = zscore_within(dl.dropna(subset=cols).reset_index(drop=True), cols)
    rows = []
    for a, b in [(SEASONS[0], SEASONS[1]), (SEASONS[1], SEASONS[0])]:
        tr, te = d[d.Season == a], d[d.Season == b]
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=RNG)
        m.fit(tr[cols], label(tr))
        pi = permutation_importance(m, te[cols], label(te), scoring="balanced_accuracy",
                                    n_repeats=30, random_state=RNG, n_jobs=-1)
        for c, v in zip(cols, pi.importances_mean):
            idx, par = c.split("_", 1)
            rows.append({"direction": f"{a} -> {b}", "feature": c, "index": idx, "parameter": par, "importance": v})
    imp = pd.DataFrame(rows)
    imp.to_csv(RESULTS / "A4_dl_permutation_importance.csv", index=False)
    by_par = (imp.assign(pos=imp.importance.clip(lower=0))
                 .groupby(["direction", "parameter"]).pos.sum()
                 .groupby(level=0).transform(lambda x: 100 * x / x.sum())
                 .rename("share_of_positive_importance_pct").reset_index())
    by_par.to_csv(RESULTS / "A5_dl_importance_by_parameter.csv", index=False)
    return imp, by_par


if __name__ == "__main__":
    print("A: main classifier sweep")
    run_main()
    print("A: detection by date")
    run_per_das()
    print("A: importance")
    run_importance()
    print("done")
