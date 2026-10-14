"""Analysis B: UAV thermal canopy temperature under managed drought (2025-26 only; the 2024-25
MicaSense data has no thermal band).

B1  Canopy temperature by water regime on each flight date, tested with the correct split-plot
    error strata (water regime vs replication x water regime, 1 and 3 df; nitrogen, seaweed and
    interactions vs the subplot residual, 66 df), Benjamini-Hochberg across dates.
B2  Two relative stress indices that need no air temperature:
      dT_ref  = plot temperature - mean of the irrigated plots on that date
      CWSI_e  = empirical crop water stress index with statistical baselines
                (T - T_wet) / (T_dry - T_wet), T_wet / T_dry = 5th / 95th percentile of all
                plots on that date. This is a within-date relative index, not the physical CWSI.
B3  Agreement with leaf physiology measured by hand (gsw, leaf temperature, PhiPS2, ETR).
B4  Correlation with grain and biological yield by date, thermal against NDVI.
B5  Drought vs irrigated classification by date: thermal alone, multispectral alone, both.
B6  Grain-yield prediction with and without the thermal series.
C   (supplementary) UAV canopy height (dCHM) against ruler plant height on matched dates.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

from common import DATA, RESULTS, load_2526_native

warnings.filterwarnings("ignore")
RNG = 0
SEASON = "2025-26"


def load():
    ts = load_2526_native()
    design = pd.read_csv(DATA / "Plot_Design.csv")
    design = design[design.Season == SEASON][["Plot_UID", "N_Level", "Seaweed_Level"]]
    harvest = pd.read_csv(DATA / "Harvest_PlotLevel.csv")
    harvest = harvest[harvest.Season == SEASON][["Plot_UID", "Grain_Yield_t_ha", "Biological_Yield_t_ha", "Harvest_Index"]]
    phys = pd.read_csv(DATA / "Physiology_PlotLevel.csv")
    phys = phys[phys.Season == SEASON][["Plot_UID", "gsw", "Tleaf", "PhiPS2", "ETR"]]
    ts = ts.merge(design, on="Plot_UID").merge(harvest, on="Plot_UID").merge(phys, on="Plot_UID")
    return ts


# ---------------------------------------------------------------- B1 / B2
def splitplot_tests(d, y):
    m = smf.ols(f"{y} ~ C(Replication)*C(Water_Regime) + C(N_Level)*C(Seaweed_Level)*C(Water_Regime)", d).fit()
    a = anova_lm(m, typ=1)
    ms = a["sum_sq"] / a["df"]
    out = {}
    main_err = "C(Replication):C(Water_Regime)"
    f = ms["C(Water_Regime)"] / ms[main_err]
    out["Water_F"], out["Water_p"] = f, stats.f.sf(f, 1, a.loc[main_err, "df"])
    for term, name in [("C(N_Level)", "N"), ("C(Seaweed_Level)", "Seaweed"),
                       ("C(N_Level):C(Water_Regime)", "NxWater"), ("C(N_Level):C(Seaweed_Level)", "NxSeaweed")]:
        f = ms[term] / ms["Residual"]
        out[f"{name}_F"], out[f"{name}_p"] = f, stats.f.sf(f, a.loc[term, "df"], a.loc["Residual", "df"])
    out["resid_df"] = a.loc["Residual", "df"]
    return out


def add_relative_indices(ts):
    ts = ts.copy()
    irrigated_mean = ts[ts.Water_Regime == "I"].groupby("DAS")["Thermal"].mean()
    ts["dT_ref"] = ts["Thermal"] - ts["DAS"].map(irrigated_mean)
    lo = ts.groupby("DAS")["Thermal"].transform(lambda t: t.quantile(0.05))
    hi = ts.groupby("DAS")["Thermal"].transform(lambda t: t.quantile(0.95))
    ts["CWSI_e"] = ((ts["Thermal"] - lo) / (hi - lo)).clip(0, 1)
    return ts


def b1_b2(ts):
    rows = []
    for das, g in ts.groupby("DAS"):
        mp = g.groupby(["Replication", "Water_Regime"])["Thermal"].mean().unstack()  # main-plot means
        diff = mp["D"] - mp["I"]
        t_crit = stats.t.ppf(0.975, len(diff) - 1)
        row = {"DAS": das,
               "T_irrigated_mean": g.loc[g.Water_Regime == "I", "Thermal"].mean(),
               "T_drought_mean": g.loc[g.Water_Regime == "D", "Thermal"].mean(),
               "dT_D_minus_I": diff.mean(),
               "dT_ci_low": diff.mean() - t_crit * diff.std(ddof=1) / np.sqrt(len(diff)),
               "dT_ci_high": diff.mean() + t_crit * diff.std(ddof=1) / np.sqrt(len(diff)),
               "reps_with_D_warmer": int((diff > 0).sum()),
               "CWSI_e_irrigated": g.loc[g.Water_Regime == "I", "CWSI_e"].mean(),
               "CWSI_e_drought": g.loc[g.Water_Regime == "D", "CWSI_e"].mean(),
               "plot_sd_all": g["Thermal"].std()}
        row.update(splitplot_tests(g, "Thermal"))
        rows.append(row)
    res = pd.DataFrame(rows)
    for term in ["Water", "N", "Seaweed", "NxWater", "NxSeaweed"]:
        res[f"{term}_q"] = multipletests(res[f"{term}_p"], method="fdr_bh")[1]
    res.to_csv(RESULTS / "B1_thermal_by_date_splitplot.csv", index=False)

    season = (ts.groupby(["Plot_UID", "Water_Regime", "Replication", "N_Level", "Seaweed_Level"])
                [["Thermal", "dT_ref", "CWSI_e"]].mean().reset_index())
    srow = []
    for v in ["Thermal", "dT_ref", "CWSI_e"]:
        r = {"variable": f"season mean {v}",
             "irrigated": season.loc[season.Water_Regime == "I", v].mean(),
             "drought": season.loc[season.Water_Regime == "D", v].mean()}
        r.update(splitplot_tests(season, v))
        srow.append(r)
    pd.DataFrame(srow).to_csv(RESULTS / "B2_season_mean_stress_indices.csv", index=False)
    return res, season


# ---------------------------------------------------------------- B3 / B4
def corr_pair(x, y, regime):
    ok = x.notna() & y.notna()
    x, y, regime = x[ok], y[ok], regime[ok]
    r, p = stats.pearsonr(x, y)
    # within-regime: centre both variables on their regime means so the drought-vs-irrigated
    # contrast cannot create the correlation on its own
    xc = x - x.groupby(regime).transform("mean")
    yc = y - y.groupby(regime).transform("mean")
    rw, pw = stats.pearsonr(xc, yc)
    return r, p, rw, pw


def b3_b4(ts, season):
    rows = []
    targets = {"gsw": "physiology", "Tleaf": "physiology", "PhiPS2": "physiology", "ETR": "physiology",
               "Grain_Yield_t_ha": "yield", "Biological_Yield_t_ha": "yield"}
    for das, g in ts.groupby("DAS"):
        for pred in ["Thermal", "NDVI"]:
            for tgt, kind in targets.items():
                r, p, rw, pw = corr_pair(g[pred], g[tgt], g["Water_Regime"])
                rows.append({"DAS": das, "predictor": pred, "target": tgt, "kind": kind,
                             "r_pooled": r, "p_pooled": p, "r_within_regime": rw, "p_within_regime": pw})
    res = pd.DataFrame(rows)
    for col in ["p_pooled", "p_within_regime"]:
        res[col.replace("p_", "q_")] = res.groupby(["predictor", "target"])[col].transform(
            lambda p: multipletests(p, method="fdr_bh")[1])
    res.to_csv(RESULTS / "B3_B4_correlations_by_date.csv", index=False)

    # season-mean stress indices against physiology and yield
    extra = ts.groupby("Plot_UID")[["gsw", "Tleaf", "PhiPS2", "ETR", "Grain_Yield_t_ha", "Biological_Yield_t_ha"]].first()
    s = season.set_index("Plot_UID").join(extra)
    srows = []
    for pred in ["Thermal", "dT_ref", "CWSI_e"]:
        for tgt in extra.columns:
            r, p, rw, pw = corr_pair(s[pred], s[tgt], s["Water_Regime"])
            srows.append({"predictor": f"season mean {pred}", "target": tgt, "r_pooled": r, "p_pooled": p,
                          "r_within_regime": rw, "p_within_regime": pw})
    pd.DataFrame(srows).to_csv(RESULTS / "B3_season_mean_vs_physiology_yield.csv", index=False)

    # does the leaf physiology itself show the drought? (split-plot tests)
    phys = ts.drop_duplicates("Plot_UID")
    prow = []
    for v in ["gsw", "Tleaf", "PhiPS2", "ETR", "Grain_Yield_t_ha", "Biological_Yield_t_ha", "Harvest_Index"]:
        r = {"variable": v, "irrigated": phys.loc[phys.Water_Regime == "I", v].mean(),
             "drought": phys.loc[phys.Water_Regime == "D", v].mean()}
        r.update(splitplot_tests(phys, v))
        prow.append(r)
    pd.DataFrame(prow).to_csv(RESULTS / "B3_physiology_water_effect.csv", index=False)
    return res


# ---------------------------------------------------------------- B5
def loro_classify(d, cols, model_name):
    X = d[cols].to_numpy()
    y = (d["Water_Regime"] == "D").astype(int).to_numpy()
    g = d["Replication"].to_numpy()
    pred, sc = np.zeros(len(y), int), np.zeros(len(y))
    for rep in np.unique(g):
        tr, te = g != rep, g == rep
        if model_name == "LDA (shrinkage)":
            m = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")).fit(X[tr], y[tr])
        else:
            m = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                             {"logisticregression__C": [0.01, 0.1, 1, 10]}, cv=GroupKFold(3),
                             scoring="balanced_accuracy").fit(X[tr], y[tr], groups=g[tr]).best_estimator_
        pred[te], sc[te] = m.predict(X[te]), m.decision_function(X[te])
    return balanced_accuracy_score(y, pred), roc_auc_score(y, sc)


def b5(ts):
    ms_cols = [c for c in ["blue", "green", "red", "rededge", "nir", "MGVRI", "NDVI", "MSAVI", "SAVI", "OSAVI", "ARVI", "EVI",
                           "GCI", "SIPI", "CI(rededge)", "NDRE", "CVI", "gNDVI", "bNDVI", "PI", "VARI", "ENDVI", "GDVI", "PSRI",
                           "MSRGR", "TNDGR", "MSR", "MCARI", "MTCI", "MCARI/OSAVI", "RVI", "DVI", "ARI", "ARI2", "SRPI", "NPCI",
                           "VDVI", "WDRVI", "TVI", "RDVI"] if c in ts.columns]
    rows = []
    for das, g in ts.groupby("DAS"):
        g = g.reset_index(drop=True)
        for fs, cols in [("Thermal only", ["Thermal"]), ("Multispectral", ms_cols), ("Multispectral + thermal", ms_cols + ["Thermal"])]:
            for mn in ["LDA (shrinkage)", "Logistic (L2)"]:
                ba, auc = loro_classify(g, cols, mn)
                rows.append({"DAS": das, "feature_set": fs, "model": mn, "balanced_acc": ba, "auc": auc})
    # whole season
    wide = ts.pivot_table(index=["Plot_UID", "Water_Regime", "Replication"], columns="DAS",
                          values=ms_cols + ["Thermal"]).reset_index()
    wide.columns = ["_".join(map(str, c)).rstrip("_") if isinstance(c, tuple) else c for c in wide.columns]
    th = [c for c in wide.columns if c.startswith("Thermal_")]
    ms = [c for c in wide.columns if c.split("_")[0] in ms_cols and c not in th]
    for fs, cols in [("Thermal only", th), ("Multispectral", ms), ("Multispectral + thermal", ms + th)]:
        for mn in ["LDA (shrinkage)", "Logistic (L2)"]:
            ba, auc = loro_classify(wide, cols, mn)
            rows.append({"DAS": "all dates", "feature_set": fs, "model": mn, "balanced_acc": ba, "auc": auc})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "B5_classification_thermal_vs_multispectral.csv", index=False)
    return res, wide, ms, th


# ---------------------------------------------------------------- B6
def b6(ts, wide, ms, th):
    y = ts.drop_duplicates("Plot_UID").set_index("Plot_UID").loc[wide["Plot_UID"], "Grain_Yield_t_ha"].to_numpy()
    g = wide["Replication"].to_numpy()
    six = [c for c in ms if c.split("_")[0] in ["NDVI", "NDRE", "CI(rededge)", "GCI", "gNDVI", "RVI"]]
    sets = {"Thermal series (10 dates)": th, "6 VIs series": six, "6 VIs + thermal": six + th,
            "All bands + VIs series": ms, "All bands + VIs + thermal": ms + th}
    rows = []
    for name, cols in sets.items():
        X = wide[cols].to_numpy()
        for mn in ["Ridge", "Random forest"]:
            for scheme in ["LORO", "Repeated 5-fold (5x)"]:
                splits = ([(np.where(g != r)[0], np.where(g == r)[0]) for r in np.unique(g)] if scheme == "LORO"
                          else list(RepeatedKFold(n_splits=5, n_repeats=5, random_state=RNG).split(X)))
                preds = []
                for tr, te in splits:
                    m = (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))) if mn == "Ridge"
                         else RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=RNG))
                    m.fit(X[tr], y[tr])
                    preds.append(pd.Series(m.predict(X[te]), index=te))
                if scheme == "LORO":
                    p = pd.concat(preds).sort_index().to_numpy()
                    r2 = r2_score(y, p)
                else:
                    p = pd.concat(preds).groupby(level=0).mean().sort_index().to_numpy()
                    r2 = r2_score(y, p)
                rows.append({"feature_set": name, "n_features": len(cols), "model": mn, "scheme": scheme,
                             "r2": r2, "rmse_t_ha": float(np.sqrt(np.mean((y - p) ** 2)))})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "B6_yield_prediction_with_thermal.csv", index=False)
    return res


# ---------------------------------------------------------------- C
def c_chm(ts):
    ground = pd.read_csv(DATA / "Ground_TimeSeries.csv")
    ground = ground[ground.Season == SEASON][["Plot_UID", "DAS", "Plant_height_cm"]]
    d = ts[["Plot_UID", "DAS", "Water_Regime", "dchm"]].merge(ground, on=["Plot_UID", "DAS"]).dropna()
    rows = []
    for das, g in d.groupby("DAS"):
        r, p = stats.pearsonr(g["dchm"], g["Plant_height_cm"])
        rows.append({"DAS": das, "n": len(g), "r": r, "p": p,
                     "dchm_mean": g["dchm"].mean(), "height_mean_cm": g["Plant_height_cm"].mean()})
    r, p = stats.pearsonr(d["dchm"], d["Plant_height_cm"])
    rows.append({"DAS": "all dates pooled", "n": len(d), "r": r, "p": p,
                 "dchm_mean": d["dchm"].mean(), "height_mean_cm": d["Plant_height_cm"].mean()})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "C_chm_vs_plant_height.csv", index=False)
    d.to_csv(RESULTS / "C_chm_height_pairs.csv", index=False)
    return res


if __name__ == "__main__":
    ts = add_relative_indices(load())
    ts.to_csv(RESULTS / "B0_thermal_plot_by_date.csv", index=False)
    b1, season = b1_b2(ts)
    print(b1[["DAS", "T_irrigated_mean", "T_drought_mean", "dT_D_minus_I", "reps_with_D_warmer", "Water_F", "Water_p", "Water_q"]].round(3).to_string())
    b3_b4(ts, season)
    b5r, wide, ms, th = b5(ts)
    print(b5r.groupby(["DAS", "feature_set"]).balanced_acc.max().unstack().round(3).to_string())
    print(b6(ts, wide, ms, th).round(3).to_string())
    print(c_chm(ts).round(3).to_string())
