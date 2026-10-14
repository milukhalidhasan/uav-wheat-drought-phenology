"""
Build a single two-season wheat dataset from:
  Wheat_Analysis_2024/05_Dataset/Wheat_2024-25_Structured_Dataset.xlsx   (season 2024-25)
  Wheat_Analysis_2025/MegaData2025_Cleaned.xlsx                          (season 2025-26)

Produces one multi-sheet workbook plus flat CSVs ready for:
  (a) agronomic / physiological parameters   -> yield   ML
  (b) double-logistic phenology parameters   -> yield   ML
each within season, and pooled across both seasons.

No source value is altered. Everything derived is recomputed here with one set
of formulas so the two seasons are directly comparable.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

from wheat_paths import (ROOT, F24, F25, OUT, CSVD, XLSX, RESD, FIGD,
                         ML_DIR, REPORT_DOCX, check_inputs)

OUT = os.path.join(ROOT, "Combined_TwoSeason")

S1, S2 = "2024-25", "2025-26"
FLIGHT_WINDOW = {S1: (10, 108), S2: (19, 83)}   # native UAV flight windows
COMMON_WINDOW = (19, 83)                        # overlap of the two seasons
COMMON_DAS = [20, 30, 40, 50, 60, 70, 80]       # cross-season interpolation grid
LN9 = np.log(9.0)

rd = lambda f, s: pd.read_excel(f, sheet_name=s)

# T0-T11 decoded from "Wheat Research Summary 2024.docx":
# 4 nitrogen levels x 3 seaweed levels, nitrogen varying slowest (N = T // 3, S = T % 3)
N_DOSE = {0: 0, 1: 210, 2: 300, 3: 390}          # kg/ha
N_DESC = {0: "Control", 1: "Low", 2: "Medium", 3: "High"}
S_DOSE = {0: 0.0, 1: 4.0, 2: 10.0}               # % seaweed extract
S_DESC = {0: "No application", 1: "Medium", 2: "High"}


# ---------------------------------------------------------------------------
# 1. common double-logistic machinery
# ---------------------------------------------------------------------------
def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def dlogistic(t, base, amp, t1, t2, k1, k2):
    """y(t) = base + amp * [ sigma(k1 (t - t1)) - sigma(k2 (t - t2)) ]

    Both seasons were fitted with this family; only the naming differs.
      2024-25 : base = a,  amp = b (= e), t1 = d, t2 = g, k1 = c, k2 = f
      2025-26 : base = y0, amp = y1 - y0, t1 = t1, t2 = t2, k1 = k1, k2 = k2
                (y1 is the peak/plateau level, not the amplitude)
    """
    return base + amp * (_sig(k1 * (t - t1)) - _sig(k2 * (t - t2)))


def curve_stats(row):
    """Derived phenology metrics, identical formulas for both seasons."""
    base = row.Baseline
    amp = row.Amplitude
    t1 = row.t1_mid_greenup
    t2 = row.t2_mid_senescence
    k1 = row.k1_rate_greenup
    k2 = row.k2_rate_senescence

    def f(t):
        return dlogistic(t, base, amp, t1, t2, k1, k2)

    SOS, EOG = t1 - LN9 / k1, t1 + LN9 / k1
    SOD, EOS = t2 - LN9 / k2, t2 + LN9 / k2

    res = minimize_scalar(lambda t: -f(t),
                          bounds=(min(SOS, t1) - 5.0, max(EOS, t2) + 5.0),
                          method="bounded")
    POS, peak = float(res.x), float(-res.fun)

    lo_n, hi_n = FLIGHT_WINDOW[row.Season]
    lo_c, hi_c = COMMON_WINDOW
    auc_native = quad(f, lo_n, hi_n, limit=200)[0]
    auc_common = quad(f, lo_c, hi_c, limit=200)[0]
    auc_pheno = quad(f, SOS, EOS, limit=200)[0]
    stages = [SOS, EOG, POS, SOD, EOS]

    return pd.Series({
        "SOS": SOS, "EOG": EOG, "POS": POS, "SOD": SOD, "EOS": EOS,
        "Peak_value": peak,
        "Greenup_Duration": EOG - SOS,
        "Plateau_Duration": SOD - EOG,
        "Senescence_Duration": EOS - SOD,
        "Green_Duration": t2 - t1,
        "LOS": EOS - SOS,
        "Rate_greenup": amp * k1 / 4.0,
        "Rate_senescence": amp * k2 / 4.0,
        "AUC_native": auc_native,
        "AUC_common_19_83": auc_common,
        "AUC_above_base_common": auc_common - base * (hi_c - lo_c),
        "AUC_pheno_SOS_EOS": auc_pheno,
        "AUC_common_per_day": auc_common / (hi_c - lo_c),
        "n_stages_outside_common": int(sum(not (lo_c <= v <= hi_c) for v in stages)),
        "n_stages_outside_native": int(sum(not (lo_n <= v <= hi_n) for v in stages)),
    })


# ---------------------------------------------------------------------------
# 2. harmonise each block
# ---------------------------------------------------------------------------
def season_cols(df, season):
    df = df.copy()
    sid = 1 if season == S1 else 2
    df.insert(0, "Season", season)
    df.insert(1, "Season_ID", sid)
    if "Plot_ID" in df.columns:
        df.insert(2, "Plot_UID", "S%d_" % sid + df["Plot_ID"].astype(str))
    return df


def build_design():
    d24 = rd(F24, "Plot_Design").rename(columns={"Plot": "Field_Plot_No",
                                                 "dCHM_relative": "dCHM_relative_plot"})
    d24 = season_cols(d24, S1)
    d25 = rd(F25, "Harvest_PlotLevel")[["Plot_ID", "Water_Regime", "Treatment", "Replication"]].copy()
    d25["Field_Plot_No"] = np.nan
    d25["dCHM_relative_plot"] = np.nan
    d25 = season_cols(d25, S2)
    out = pd.concat([d24, d25], ignore_index=True)
    out["Treatment_Num"] = out["Treatment"].str.replace("T", "", regex=False).astype(int)
    out["Water_Regime_Label"] = out["Water_Regime"].map({"D": "Managed drought", "I": "Irrigated"})

    n_idx = out["Treatment_Num"] // 3
    s_idx = out["Treatment_Num"] % 3
    out["N_Level"] = "N" + n_idx.astype(str)
    out["N_kg_ha"] = n_idx.map(N_DOSE)
    out["N_Desc"] = n_idx.map(N_DESC)
    out["Seaweed_Level"] = "S" + s_idx.astype(str)
    out["Seaweed_pct"] = s_idx.map(S_DOSE)
    out["Seaweed_Desc"] = s_idx.map(S_DESC)

    cols = ["Season", "Season_ID", "Plot_UID", "Plot_ID", "Field_Plot_No", "Water_Regime",
            "Water_Regime_Label", "Treatment", "Treatment_Num", "N_Level", "N_kg_ha", "N_Desc",
            "Seaweed_Level", "Seaweed_pct", "Seaweed_Desc", "Replication", "dCHM_relative_plot"]
    return out[cols].sort_values(["Season_ID", "Plot_ID"]).reset_index(drop=True)


HARVEST_MAP_25 = {
    "1000_Seed_Weight_(g)": "TSW_g",
    "Fresh_Biomass(t/ha)": "Fresh_Biomass_t_ha",
    "Biological_Yield(t/ha)": "Biological_Yield_t_ha",
    "Grain_Yield(t/ha)": "Grain_Yield_t_ha",
    "Harvest_Index(HI)": "Harvest_Index",
    "H2O2 (umol/g FW)": "H2O2_umol_per_gFW",
}

HARVEST_ORDER = ["Season", "Season_ID", "Plot_UID", "Plot_ID", "Water_Regime", "Treatment",
                 "Replication", "Plant_Length_cm", "Root_Length_cm", "Tiller_No",
                 "Spike_Length_cm", "Total_Grain_per_Plant", "TSW_g", "Fresh_Biomass_t_ha",
                 "Biological_Yield_t_ha", "Grain_Yield_t_ha", "Harvest_Index",
                 "H2O2_umol_per_gFW", "MDA_nmol_per_gFW"]


def build_harvest():
    h24 = rd(F24, "Harvest_PlotLevel").drop(columns=["Plot"]).pipe(season_cols, S1)
    h25 = rd(F25, "Harvest_PlotLevel").rename(columns=HARVEST_MAP_25)
    # the 2025-26 harvest sheet also carries leaf physiology; split it out
    phys25 = h25[["Plot_ID", "gsw", "PhiPS2", "ETR", "Tleaf"]].copy()
    h25 = h25.drop(columns=["gsw", "PhiPS2", "ETR", "Tleaf"]).pipe(season_cols, S2)
    out = pd.concat([h24, h25], ignore_index=True)
    for c in HARVEST_ORDER:
        if c not in out.columns:
            out[c] = np.nan
    out = out[HARVEST_ORDER].sort_values(["Season_ID", "Plot_ID"]).reset_index(drop=True)
    return out, phys25


def build_physiology(phys25):
    """2024-25 measured a full gas-exchange panel at 80 and 88 DAS; 2025-26 has a
    single set in the harvest sheet. The two dates are averaged so the four
    shared variables line up, and the per-date values are kept alongside."""
    lp = rd(F24, "Leaf_Physiology")
    idcols = ["Plot_ID", "Water_Regime", "Replication", "Treatment", "DAS"]
    vars24 = [c for c in lp.columns if c not in idcols]
    mean24 = lp.groupby("Plot_ID")[vars24].mean().reset_index()
    wide = lp.pivot_table(index="Plot_ID", columns="DAS", values=["gsw", "PhiPS2", "ETR", "Tleaf"])
    wide.columns = ["%s_DAS%d" % (v, d) for v, d in wide.columns]
    p24 = mean24.merge(wide.reset_index(), on="Plot_ID").pipe(season_cols, S1)
    p24["Physiology_Source"] = "mean of 80 and 88 DAS"

    p25 = phys25.copy().pipe(season_cols, S2)
    p25["Physiology_Source"] = "single measurement (harvest sheet)"

    out = pd.concat([p24, p25], ignore_index=True)
    lead = ["Season", "Season_ID", "Plot_UID", "Plot_ID", "Physiology_Source",
            "gsw", "PhiPS2", "ETR", "Tleaf"]
    rest = [c for c in out.columns if c not in lead]
    return out[lead + rest].sort_values(["Season_ID", "Plot_ID"]).reset_index(drop=True)


GROUND_COLS = ["Season", "Season_ID", "Plot_UID", "Plot_ID", "Water_Regime", "Replication",
               "Treatment", "Measure_Date", "DAS", "Growth_Stage", "Chlorophyll",
               "Plant_height_cm", "Soil_Temp_C", "Soil_Moisture_pct"]


def build_ground():
    g24 = rd(F24, "Ground_TimeSeries").rename(columns={"Plant_height": "Plant_height_cm",
                                                       "Soil_Temp": "Soil_Temp_C",
                                                       "Soil_Moisture": "Soil_Moisture_pct"})
    g24["Measure_Date"] = pd.NaT
    g24["Growth_Stage"] = np.nan
    g24 = season_cols(g24, S1)
    g25 = rd(F25, "Ground_TimeSeries").rename(columns={"DOY": "Measure_Date"})
    g25["Soil_Moisture_pct"] = np.nan
    g25 = season_cols(g25, S2)
    out = pd.concat([g24[GROUND_COLS], g25[GROUND_COLS]], ignore_index=True)
    return out.sort_values(["Season_ID", "Plot_ID", "DAS"]).reset_index(drop=True)


def last_valid(s):
    s = s.dropna()
    return s.iloc[-1] if len(s) else np.nan


def summarise_ground(g):
    """Plot-level summary of the ground campaign. Sampling dates differ between
    seasons, so these are campaign summaries, not matched-date values."""
    return (g.sort_values("DAS")
             .groupby(["Season", "Season_ID", "Plot_UID", "Plot_ID"], as_index=False)
             .agg(Chlorophyll_mean=("Chlorophyll", "mean"),
                  Chlorophyll_max=("Chlorophyll", "max"),
                  Chlorophyll_last=("Chlorophyll", last_valid),
                  Plant_height_max_cm=("Plant_height_cm", "max"),
                  Plant_height_mean_cm=("Plant_height_cm", "mean"),
                  Soil_Temp_mean_C=("Soil_Temp_C", "mean"),
                  Soil_Moisture_mean_pct=("Soil_Moisture_pct", "mean"),
                  n_ground_dates=("DAS", "nunique")))


SPEC_LEAD = ["Season", "Season_ID", "Plot_UID", "Plot_ID", "Water_Regime", "Replication",
             "Treatment", "Flight_Date", "DAS"]
BANDS = ["blue", "green", "red", "rededge", "nir", "Thermal", "dchm"]
SEASON_ONLY_SPECTRAL = ["Thermal", "dchm", "SR"]


def build_spectral():
    s24 = rd(F24, "Spectral_TimeSeries").rename(columns={"DOY": "Flight_Date"})
    s24["Growth_Stage"] = np.nan
    s24["Thermal"] = np.nan
    s24["dchm"] = np.nan
    s24 = season_cols(s24, S1)

    s25 = rd(F25, "Spectral_TimeSeries").rename(columns={"DOY": "Flight_Date"})
    s25 = s25.drop(columns=["L_coef", "C1_coef", "C2_coef"])  # constant, no information
    s25["SR"] = np.nan
    s25 = season_cols(s25, S2)

    meta = set(SPEC_LEAD + BANDS + ["Growth_Stage"])
    vis = [c for c in s25.columns if c not in meta and c != "SR"]
    vis = [c for c in vis if c in s24.columns] + ["SR"]

    cols = SPEC_LEAD + ["Growth_Stage"] + BANDS + vis
    out = pd.concat([s24.reindex(columns=cols), s25.reindex(columns=cols)], ignore_index=True)
    out["Flight_Date"] = pd.to_datetime(out["Flight_Date"])
    out.insert(9, "DOY_calendar", out["Flight_Date"].dt.dayofyear)
    out.insert(10, "Flight_No", out.groupby(["Season", "Plot_ID"])["DAS"].rank(method="dense").astype(int))
    return out.sort_values(["Season_ID", "Plot_ID", "DAS"]).reset_index(drop=True), vis


def interp_common(spec, vis):
    """Linear interpolation of every shared band/index onto a DAS grid that both
    seasons actually cover (19-83 DAS overlap)."""
    value_cols = [c for c in BANDS + vis if c not in SEASON_ONLY_SPECTRAL]
    keys = ["Season", "Season_ID", "Plot_UID", "Plot_ID", "Water_Regime", "Replication", "Treatment"]
    rows = []
    for key, g in spec.groupby(keys, sort=False):
        g = g.sort_values("DAS")
        x = g["DAS"].to_numpy(float)
        rec = dict(zip(keys, key))
        for v in value_cols:
            y = g[v].to_numpy(float)
            ok = ~np.isnan(y)
            for d in COMMON_DAS:
                rec["%s_DAS%d" % (v, d)] = np.interp(d, x[ok], y[ok]) if ok.sum() >= 2 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["Season_ID", "Plot_ID"]).reset_index(drop=True)


PHENO_KEEP = ["Season", "Season_ID", "Plot_UID", "Plot_ID", "Water_Regime", "Replication",
              "Treatment", "Index", "Param_Notation_Source", "Baseline", "Amplitude",
              "Peak_asymptote", "t1_mid_greenup", "t2_mid_senescence", "k1_rate_greenup",
              "k2_rate_senescence", "R2", "RMSE", "Params_At_Bound", "AUC_source"]


def build_phenology():
    p24 = rd(F24, "Phenology_Parameters")
    p24 = p24.assign(Baseline=p24["a"],
                     Amplitude=p24["b"],
                     Peak_asymptote=p24["a"] + p24["b"],
                     t1_mid_greenup=p24["d"],
                     t2_mid_senescence=p24["g"],
                     k1_rate_greenup=p24["c"],
                     k2_rate_senescence=p24["f"],
                     AUC_source=p24["AUC"],
                     Params_At_Bound=p24["n_params_at_bound"])
    p24 = season_cols(p24, S1)
    p24["Param_Notation_Source"] = "a,b,c,d,e,f,g (b = e)"

    p25 = rd(F25, "Phenology_Parameters")
    p25 = p25.assign(Baseline=p25["y0"],
                     Amplitude=p25["y1"] - p25["y0"],
                     Peak_asymptote=p25["y1"],
                     t1_mid_greenup=p25["t1"],
                     t2_mid_senescence=p25["t2"],
                     k1_rate_greenup=p25["k1"],
                     k2_rate_senescence=p25["k2"],
                     AUC_source=p25["AUC"])
    p25 = season_cols(p25, S2)
    p25["Param_Notation_Source"] = "y0,y1,t1,t2,k1,k2 (y1 = peak)"

    out = pd.concat([p24[PHENO_KEEP], p25[PHENO_KEEP]], ignore_index=True)
    out = pd.concat([out, out.apply(curve_stats, axis=1)], axis=1)
    out["Fit_Quality_Flag"] = np.where(out.R2 < 0.80, "poor",
                                       np.where((out.R2 < 0.90) | (out.Params_At_Bound > 0),
                                                "check", "ok"))
    return out.sort_values(["Season_ID", "Index", "Plot_ID"]).reset_index(drop=True)


PHENO_ML_PARAMS = ["Baseline", "Amplitude", "Peak_asymptote", "Peak_value",
                   "t1_mid_greenup", "t2_mid_senescence", "k1_rate_greenup", "k2_rate_senescence",
                   "SOS", "EOG", "POS", "SOD", "EOS",
                   "Greenup_Duration", "Plateau_Duration", "Senescence_Duration",
                   "Green_Duration", "LOS", "Rate_greenup", "Rate_senescence",
                   "AUC_common_19_83", "AUC_above_base_common", "AUC_pheno_SOS_EOS", "R2"]


def widen_phenology(ph):
    keys = ["Season", "Season_ID", "Plot_UID", "Plot_ID", "Water_Regime", "Replication", "Treatment"]
    w = ph.pivot_table(index=keys, columns="Index", values=PHENO_ML_PARAMS)
    w.columns = ["%s_%s" % (ix.replace("(", "").replace(")", ""), par) for par, ix in w.columns]
    w = w.reset_index()
    return w[keys + sorted(c for c in w.columns if c not in keys)] \
            .sort_values(["Season_ID", "Plot_ID"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. ML-ready tables
# ---------------------------------------------------------------------------
YIELD_TARGETS = ["Grain_Yield_t_ha", "Biological_Yield_t_ha", "Harvest_Index"]
YIELD_COMPONENTS = ["Tiller_No", "Spike_Length_cm", "Total_Grain_per_Plant", "TSW_g"]
AGRONOMIC = ["Plant_Length_cm", "Root_Length_cm", "Fresh_Biomass_t_ha"]
STRESS = ["H2O2_umol_per_gFW", "MDA_nmol_per_gFW"]
PHYSIOLOGY = ["gsw", "PhiPS2", "ETR", "Tleaf"]
GROUND_SUMMARY = ["Chlorophyll_mean", "Chlorophyll_max", "Chlorophyll_last",
                  "Plant_height_max_cm", "Plant_height_mean_cm",
                  "Soil_Temp_mean_C", "Soil_Moisture_mean_pct"]
# present in only one season -> excluded from the pooled cross-season feature set
SEASON_SPECIFIC = ["MDA_nmol_per_gFW", "Soil_Moisture_mean_pct", "dCHM_relative_plot"]

# experimental factors, decoded from the treatment codes; usable directly as predictors
DESIGN_FEATURES = ["N_Level", "N_kg_ha", "Seaweed_Level", "Seaweed_pct"]

KEYS = ["Season", "Season_ID", "Plot_UID", "Plot_ID", "Water_Regime", "Treatment"] \
       + DESIGN_FEATURES + ["Replication"]


def build_ml_agronomic(design, harvest, phys, gsum):
    df = (design.drop(columns=["Field_Plot_No", "Water_Regime_Label", "Treatment_Num",
                               "N_Desc", "Seaweed_Desc"])
                .merge(harvest.drop(columns=["Water_Regime", "Treatment", "Replication",
                                             "Season_ID", "Season"]),
                       on=["Plot_UID", "Plot_ID"], how="left", validate="1:1")
                .merge(phys[["Plot_UID"] + PHYSIOLOGY], on="Plot_UID", how="left", validate="1:1")
                .merge(gsum[["Plot_UID"] + GROUND_SUMMARY], on="Plot_UID", how="left",
                       validate="1:1"))
    feats = AGRONOMIC + YIELD_COMPONENTS + STRESS + PHYSIOLOGY + GROUND_SUMMARY + ["dCHM_relative_plot"]
    cols = KEYS + [c for c in feats if c in df.columns] + YIELD_TARGETS
    return df[cols].sort_values(["Season_ID", "Plot_ID"]).reset_index(drop=True)


def build_ml_phenology(phw, harvest, design):
    y = harvest[["Plot_UID"] + YIELD_TARGETS]
    df = (phw.merge(design[["Plot_UID"] + DESIGN_FEATURES], on="Plot_UID", how="left",
                    validate="1:1")
             .merge(y, on="Plot_UID", how="left", validate="1:1"))
    rest = [c for c in df.columns if c not in KEYS + YIELD_TARGETS]
    return df[KEYS + rest + YIELD_TARGETS].sort_values(["Season_ID", "Plot_ID"]).reset_index(drop=True)


def drop_season_specific(df, min_frac=0.5, label=""):
    """Pooled cross-season view: keep only columns populated in both seasons.

    A column must be at least `min_frac` populated *within each season*. Testing
    only .notna().any() would let a column carrying a single value in one season
    pass as if it were fully measured there.
    """
    out = df.drop(columns=[c for c in SEASON_SPECIFIC if c in df.columns])
    keep, thin = [], []
    for c in out.columns:
        if c in KEYS:
            keep.append(c)
            continue
        frac = out.groupby("Season")[c].apply(lambda s: s.notna().mean())
        if (frac >= min_frac).all():
            keep.append(c)
            if (frac < 1.0).any():
                thin.append("%s (%s)" % (c, ", ".join("%s %.0f%%" % (s, f * 100)
                                                      for s, f in frac.items())))
    if thin:
        print("  %s: kept but incomplete -> %s" % (label or "cross-season", "; ".join(thin)))
    return out[keep]


# ---------------------------------------------------------------------------
# 4. documentation tables
# ---------------------------------------------------------------------------
def build_readme():
    rows = [
        ("DATASET", "Two-season wheat UAV + agronomy dataset, Gazipur Agricultural University", ""),
        ("Variety", "BWMRI Gom-1", "same both seasons"),
        ("Design", "Split plot: water regime on main plots, 12 treatments on subplots, 4 replications",
         "96 plots per season, 192 plot-season records"),
        ("Water regimes", "D = managed drought, I = irrigated", "48 plots each per season"),
        ("Treatments", "T0-T11, full factorial of 4 nitrogen levels x 3 seaweed extract levels",
         "decoded from Wheat Research Summary 2024.docx"),
        ("Nitrogen levels", "N0 = 0, N1 = 210, N2 = 300, N3 = 390 kg/ha", "column N_kg_ha"),
        ("Seaweed levels", "S0 = 0 %, S1 = 4 %, S2 = 10 %", "column Seaweed_pct"),
        ("Treatment decoding", "N index = Treatment_Num // 3, S index = Treatment_Num mod 3",
         "T0 = N0S0, T1 = N0S1, T2 = N0S2, T3 = N1S0, ... T11 = N3S2"),
        ("Season 2024-25", "11 UAV flights, 10-108 DAS, 25 Nov 2024 to 03 Mar 2025", "source: Wheat_2024-25_Structured_Dataset.xlsx"),
        ("Season 2025-26", "10 UAV flights, 19-83 DAS, 22 Dec 2025 to 24 Feb 2026", "source: MegaData2025_Cleaned.xlsx"),
        ("Keys", "Plot_UID = S<Season_ID>_<Plot_ID>, unique across both seasons", "Plot_ID repeats between seasons"),
        ("", "", ""),
        ("SHEET", "CONTENTS", "ROWS"),
        ("Plot_Design", "One row per plot-season: identifiers, water regime, treatment, replication", "192"),
        ("Harvest_PlotLevel", "Harmonised yield, yield components, biomass, H2O2, MDA", "192"),
        ("Physiology_PlotLevel", "Gas exchange and fluorescence, harmonised to four shared variables", "192"),
        ("Ground_TimeSeries", "Chlorophyll, plant height, soil temperature, soil moisture, long format", "1248"),
        ("Ground_PlotSummary", "Plot-level campaign summaries of the ground series", "192"),
        ("Spectral_TimeSeries", "One row per plot per flight, bands and vegetation indices, long format", "2016"),
        ("Spectral_Interp_CommonDAS", "Every shared band/index interpolated to 20-80 DAS, one row per plot-season", "192"),
        ("Phenology_Parameters", "Double-logistic parameters in one common notation plus derived stages", "1152"),
        ("Phenology_Wide", "Phenology parameters pivoted to one row per plot-season", "192"),
        ("ML_Agronomic_vs_Yield", "Agronomic / physiological features + yield targets, all columns", "192"),
        ("ML_Phenology_vs_Yield", "Double-logistic features + yield targets, all columns", "192"),
        ("ML_Agronomic_CrossSeason", "Same, restricted to variables present in both seasons", "192"),
        ("ML_Phenology_CrossSeason", "Same, restricted to variables present in both seasons", "192"),
        ("Coverage_Matrix", "Which variables are populated in which season", ""),
        ("Data_Dictionary", "Every column, its role and its units", ""),
        ("Known_Issues", "Things to check before trusting a result", ""),
        ("Index_Screening_2024", "2024-25 index screening table, carried over unchanged", "35"),
        ("", "", ""),
        ("HARMONISED MODEL", "y(t) = Baseline + Amplitude * [ sigma(k1 (t - t1)) - sigma(k2 (t - t2)) ]", ""),
        ("2024-25 mapping", "Baseline = a, Amplitude = b (= e), t1 = d, t2 = g, k1 = c, k2 = f", "verified against source R2"),
        ("2025-26 mapping", "Baseline = y0, Amplitude = y1 - y0, t1 = t1, t2 = t2, k1 = k1, k2 = k2", "y1 is the peak, not the amplitude"),
        ("", "", ""),
        ("DERIVED STAGES", "recomputed here for both seasons with the same formulas", ""),
        ("SOS / EOG", "t1 -/+ ln(9)/k1", "DAS"),
        ("SOD / EOS", "t2 -/+ ln(9)/k2", "DAS"),
        ("POS / Peak_value", "numerical maximum of the fitted curve", "DAS / index units"),
        ("Green_Duration", "t2 - t1", "days"),
        ("Plateau_Duration", "SOD - EOG", "days"),
        ("LOS", "EOS - SOS", "days"),
        ("Rate_greenup / Rate_senescence", "Amplitude * k / 4, slope at the inflection", "index units per day"),
        ("", "", ""),
        ("AUC VARIANTS", "the two seasons were flown over different windows, so AUC needs care", ""),
        ("AUC_native", "integral over each season's own flight window (10-108 vs 19-83 DAS)", "reproduces the source AUC"),
        ("AUC_common_19_83", "integral over 19-83 DAS in BOTH seasons", "USE THIS for cross-season work"),
        ("AUC_above_base_common", "AUC_common minus Baseline * 64, removes the baseline offset", "recommended"),
        ("AUC_pheno_SOS_EOS", "integral between the plot's own SOS and EOS", "phenology-normalised"),
        ("", "", ""),
        ("CROSS-SEASON TIME ALIGNMENT", "flight dates do not coincide between seasons", ""),
        ("Spectral_Interp_CommonDAS", "linear interpolation to 20, 30, 40, 50, 60, 70, 80 DAS", "inside both seasons' coverage"),
        ("Phenology features", "already time-normalised by construction, use them directly", ""),
    ]
    return pd.DataFrame(rows, columns=["Item", "Description", "Notes"])


def build_known_issues():
    rows = [
        ("Grain yield in the 2025-26 phenology sheet (RESOLVED)",
         "MegaData2025_Cleaned.xlsx Phenology_Parameters carries a Grain_Yield(t/ha) column that "
         "disagrees with its own Harvest_PlotLevel sheet for 9 of 96 plots "
         "(DR1T1, DR2T2, DR3T10, DR3T2, DR4T1, DR4T10, DR4T11, IR1T11 by exactly 1.00 t/ha; "
         "IR2T4 by exactly 3.00 t/ha, always higher in the phenology sheet). "
         "RESOLVED by the identity HI = grain yield / biological yield, all three columns "
         "being in the harvest sheet: for all 9 disputed plots the harvest value reproduces "
         "HI x biological yield to within 0.009 t/ha, the same precision as the 87 undisputed "
         "plots (median 0.003), while the phenology-sheet value is off by exactly 1.00 or 3.00. "
         "The harvest sheet is correct; the phenology sheet column is stale.",
         "No action needed. Grain yield is taken from Harvest_PlotLevel throughout and the "
         "phenology-sheet column is never propagated. Re-check with "
         "01_Code/verify_yield_discrepancy.py."),
        ("Different flight windows",
         "2024-25 was flown 10-108 DAS (11 flights), 2025-26 only 19-83 DAS (10 flights). "
         "Senescence is much better sampled in 2024-25.",
         "Use AUC_common_19_83 and the interpolated 20-80 DAS columns for anything that pools seasons."),
        ("Fit quality differs between seasons",
         "Mean R2 is 0.93 in 2024-25 against 0.99 in 2025-26; 241 of 576 curves in 2024-25 have at "
         "least one parameter at a bound, against 102 of 576 in 2025-26.",
         "Season is confounded with fit quality. Filter on Fit_Quality_Flag, or include it as a covariate."),
        ("Rate parameters at the 0.8 bound",
         "36 of the 2024-25 curves have k1 pinned at 0.8 per day, which makes green-up rate and "
         "SOS unreliable for those plots.",
         "Check Params_At_Bound before using SOS, EOG or Rate_greenup."),
        ("MDA is not plot-unique (2024-25)",
         "MDA_nmol_per_gFW has only 23 distinct values across 96 plots, so it looks like treatment-level "
         "means broadcast to plots.",
         "Do not treat MDA as an independent plot-level observation. Not available at all in 2025-26."),
        ("Leaf physiology is not comparable date-for-date",
         "2024-25 measured gas exchange at 80 and 88 DAS and is averaged here; 2025-26 has a single "
         "unlabelled measurement.",
         "gsw, PhiPS2, ETR and Tleaf are comparable only as season-level summaries."),
        ("Ground sensor gaps (2024-25)",
         "Soil temperature is missing at 29 and 51 DAS, soil moisture at 29, 39 and 51 DAS, "
         "plant height at 58 DAS.",
         "Ground_PlotSummary averages over whatever is present; n_ground_dates records the count."),
        ("Soil moisture only in 2024-25",
         "2025-26 recorded no soil moisture.",
         "Excluded from the cross-season feature set."),
        ("Ground campaigns do not overlap in DAS",
         "2024-25 ground data covers 17-58 DAS, 2025-26 covers 34-83 DAS.",
         "Ground summaries describe different parts of the season; interpret pooled models with care."),
        ("Design description is inconsistent between documents",
         "Wheat Research Summary 2024.docx describes an RCBD with five blocks per water condition "
         "(which would be 60 plots each), but in the same paragraph states 96 plots / 48 per condition. "
         "The 2024-25 dataset README describes a split plot with water regime on main plots. "
         "The data itself has 4 replications (R1-R4) x 12 treatments = 48 plots per water regime.",
         "The data matches 4 blocks, not 5. Settle whether the error term is split plot or RCBD before "
         "running mixed models; it does not affect the ML tables."),
        ("Collinear indices",
         "SR is identical to RVI, TCARI to MCARI, TCARI/OSAVI to MCARI/OSAVI. SR exists only in 2024-25.",
         "Drop one of each pair before fitting. SR is excluded from the interpolated table."),
        ("Thermal and dchm only in 2025-26",
         "The 2024-25 MicaSense data has no thermal band, and canopy height is plot-level "
         "(dCHM_relative_plot) rather than per flight.",
         "Excluded from the cross-season feature set."),
        ("Harvest index differs markedly between seasons",
         "Mean HI is 0.43 in 2024-25 against 0.36 in 2025-26, with very different spreads.",
         "Expect a strong season effect. Center within season, or include Season as a term."),
        ("DOY_calendar wraps at New Year",
         "Both seasons straddle 1 January, so DOY_calendar runs 330, 338, 350, 360, 6, 13, ... "
         "in 2024-25 and 356, 1, 6, ... in 2025-26. It is a genuine calendar day of year but it "
         "decreases mid-season.",
         "Never sort, plot or model on DOY_calendar. Use DAS (monotonic) or Flight_No for order, "
         "Flight_Date for real time."),
        ("SOS and EOS are often extrapolated",
         "EOS falls beyond the season's last flight for 925 of 1152 curves (median 7-8 days past, "
         "up to 128 days where k2 is small); SOS falls before the first flight for 435. "
         "AUC_pheno_SOS_EOS therefore integrates the fitted curve over stretches with no data.",
         "Prefer AUC_common_19_83 for anything quantitative. Check n_stages_outside_native before "
         "using SOS, EOS, LOS or AUC_pheno_SOS_EOS."),
        ("Ground measurement dates missing for 2024-25",
         "The 2024-25 ground sheet carries no date column, so Measure_Date is empty for that whole "
         "season while 2025-26 has real dates.",
         "Join ground data on DAS, not on Measure_Date."),
        ("Season labelling",
         "Flight dates place the second file in the 2025-26 rabi season (Dec 2025 to Feb 2026), "
         "not 2025 calendar year.",
         "Seasons are labelled 2024-25 and 2025-26 throughout."),
    ]
    return pd.DataFrame(rows, columns=["Issue", "Detail", "What to do"])


ROLE_UNITS = {
    "Grain_Yield_t_ha": ("yield_target", "t/ha", "Primary response variable"),
    "Biological_Yield_t_ha": ("yield_target", "t/ha", "Total above-ground dry yield"),
    "Harvest_Index": ("yield_target", "ratio", "Grain yield / biological yield"),
    "Tiller_No": ("yield_component", "count/plant", ""),
    "Spike_Length_cm": ("yield_component", "cm", ""),
    "Total_Grain_per_Plant": ("yield_component", "count", ""),
    "TSW_g": ("yield_component", "g", "Thousand seed weight"),
    "Plant_Length_cm": ("agronomic", "cm", "At harvest"),
    "Root_Length_cm": ("agronomic", "cm", "At harvest"),
    "Fresh_Biomass_t_ha": ("agronomic", "t/ha", ""),
    "H2O2_umol_per_gFW": ("stress_biochem", "umol/g FW", ""),
    "MDA_nmol_per_gFW": ("stress_biochem", "nmol/g FW", "2024-25 only, not plot-unique"),
    "gsw": ("physiology", "mol/m2/s", "Stomatal conductance to water"),
    "PhiPS2": ("physiology", "ratio", "Operating PSII efficiency"),
    "ETR": ("physiology", "umol/m2/s", "Electron transport rate"),
    "Tleaf": ("physiology", "degC", "Leaf temperature"),
    "Chlorophyll_mean": ("ground", "SPAD-like", "Campaign mean"),
    "Chlorophyll_max": ("ground", "SPAD-like", "Campaign maximum"),
    "Chlorophyll_last": ("ground", "SPAD-like", "Last recorded date"),
    "Plant_height_max_cm": ("ground", "cm", "Campaign maximum"),
    "Plant_height_mean_cm": ("ground", "cm", "Campaign mean"),
    "Soil_Temp_mean_C": ("ground", "degC", "Campaign mean"),
    "Soil_Moisture_mean_pct": ("ground", "%", "2024-25 only"),
    "dCHM_relative_plot": ("structure", "m", "2024-25 only, plot-level canopy height"),
    "Baseline": ("phenology", "index units", "Fitted curve baseline"),
    "Amplitude": ("phenology", "index units", "Seasonal amplitude"),
    "Peak_asymptote": ("phenology", "index units", "Baseline + amplitude"),
    "Peak_value": ("phenology", "index units", "Numerical maximum of the curve"),
    "t1_mid_greenup": ("phenology", "DAS", "Green-up inflection"),
    "t2_mid_senescence": ("phenology", "DAS", "Senescence inflection"),
    "k1_rate_greenup": ("phenology", "per day", "Green-up steepness"),
    "k2_rate_senescence": ("phenology", "per day", "Senescence steepness"),
    "SOS": ("phenology", "DAS", "Start of season"),
    "EOG": ("phenology", "DAS", "End of green-up"),
    "POS": ("phenology", "DAS", "Peak of season"),
    "SOD": ("phenology", "DAS", "Start of decline"),
    "EOS": ("phenology", "DAS", "End of season"),
    "Greenup_Duration": ("phenology", "days", "EOG - SOS"),
    "Plateau_Duration": ("phenology", "days", "SOD - EOG"),
    "Senescence_Duration": ("phenology", "days", "EOS - SOD"),
    "Green_Duration": ("phenology", "days", "t2 - t1"),
    "LOS": ("phenology", "days", "EOS - SOS"),
    "Rate_greenup": ("phenology", "index units/day", "Amplitude * k1 / 4"),
    "Rate_senescence": ("phenology", "index units/day", "Amplitude * k2 / 4"),
    "AUC_native": ("phenology", "index units x days", "Own flight window, reproduces source"),
    "AUC_common_19_83": ("phenology", "index units x days", "Cross-season comparable"),
    "AUC_above_base_common": ("phenology", "index units x days", "Baseline removed"),
    "AUC_pheno_SOS_EOS": ("phenology", "index units x days", "SOS to EOS"),
    "AUC_common_per_day": ("phenology", "index units", "AUC_common / 64"),
    "R2": ("fit_quality", "ratio", "Curve fit R2"),
    "RMSE": ("fit_quality", "index units", "Curve fit RMSE"),
    "Params_At_Bound": ("fit_quality", "count", "Parameters pinned at an optimiser bound"),
    "Season": ("key", "", "2024-25 or 2025-26"),
    "Season_ID": ("key", "", "1 or 2"),
    "Plot_UID": ("key", "", "Unique across seasons"),
    "Plot_ID": ("key", "", "Repeats between seasons"),
    "Water_Regime": ("design", "", "D drought, I irrigated"),
    "Treatment": ("design", "", "T0-T11"),
    "Treatment_Num": ("design", "", "0-11, numeric form of Treatment"),
    "N_Level": ("design", "", "N0-N3, nitrogen factor level"),
    "N_kg_ha": ("design", "kg/ha", "Nitrogen dose: 0, 210, 300, 390"),
    "N_Desc": ("design", "", "Control / Low / Medium / High"),
    "Seaweed_Level": ("design", "", "S0-S2, seaweed factor level"),
    "Seaweed_pct": ("design", "%", "Seaweed extract dose: 0, 4, 10"),
    "Seaweed_Desc": ("design", "", "No application / Medium / High"),
    "Water_Regime_Label": ("design", "", "Managed drought / Irrigated"),
    "Field_Plot_No": ("design", "", "Field plot number, 2024-25 only"),
    "Replication": ("design", "", "R1-R4"),
    "DAS": ("time", "days", "Days after sowing"),
    "Index": ("key", "", "Vegetation index the curve was fitted to"),
    # time / flight bookkeeping
    "Flight_Date": ("time", "date", "Calendar date of the UAV flight"),
    "DOY_calendar": ("time", "day of year", "Calendar day of year. WRAPS at 1 January "
                     "(e.g. 360 -> 6 mid-season), so it is NOT a usable time axis. Use DAS."),
    "Flight_No": ("time", "count", "Flight order within the season, 1 = first"),
    "Measure_Date": ("time", "date", "Ground measurement date; 2025-26 only"),
    "Growth_Stage": ("time", "", "Recorded growth stage; 2025-26 only"),
    # ground series (long format)
    "Chlorophyll": ("ground", "SPAD-like", "Per-date reading"),
    "Plant_height_cm": ("ground", "cm", "Per-date reading"),
    "Soil_Temp_C": ("ground", "degC", "Per-date reading"),
    "Soil_Moisture_pct": ("ground", "%", "Per-date reading, 2024-25 only"),
    "n_ground_dates": ("ground", "count", "Distinct ground dates behind the plot summary"),
    # extra 2024-25 gas-exchange panel (no 2025-26 counterpart)
    "gbw": ("physiology", "mol/m2/s", "Boundary-layer conductance, 2024-25 only"),
    "gtw": ("physiology", "mol/m2/s", "Total conductance to water, 2024-25 only"),
    "E_apparent": ("physiology", "mol/m2/s", "Apparent transpiration, 2024-25 only"),
    "VPleaf": ("physiology", "kPa", "Leaf vapour pressure, 2024-25 only"),
    "VPDleaf": ("physiology", "kPa", "Leaf vapour pressure deficit, 2024-25 only"),
    "H2O_leaf": ("physiology", "mmol/mol", "Leaf water vapour mole fraction, 2024-25 only"),
    "Fs": ("physiology", "relative", "Steady-state fluorescence, 2024-25 only"),
    "Fm.": ("physiology", "relative", "Light-adapted maximum fluorescence, 2024-25 only"),
    "rh_s": ("physiology", "%", "Sample relative humidity, 2024-25 only"),
    "Qamb": ("physiology", "umol/m2/s", "Ambient PAR, 2024-25 only"),
    # provenance / flags
    "Physiology_Source": ("provenance", "", "How the four shared physiology variables were derived"),
    "Param_Notation_Source": ("provenance", "", "Parameter names used in the source workbook"),
    "AUC_source": ("provenance", "index units x days", "AUC as published in the source workbook"),
    "Fit_Quality_Flag": ("fit_quality", "", "ok / check / poor, from R2 and Params_At_Bound"),
    "n_stages_outside_common": ("fit_quality", "count", "Stages falling outside 19-83 DAS"),
    "n_stages_outside_native": ("fit_quality", "count", "Stages extrapolated beyond the season's "
                                "own flight window; >0 means SOS/EOS rest on no data"),
}


def build_dictionary(tables):
    # the vegetation indices are whatever the spectral series carries beyond
    # the keys and the raw bands; anything else must be described explicitly
    spec = tables.get("Spectral_TimeSeries")
    index_names = set() if spec is None else (
        set(spec.columns) - set(SPEC_LEAD) - set(BANDS)
        - {"Growth_Stage", "DOY_calendar", "Flight_No"})
    seen, rows = set(), []
    for tname, df in tables.items():
        for c in df.columns:
            if c in seen:
                continue
            seen.add(c)
            role, unit, note = ROLE_UNITS.get(c, ("", "", ""))
            if not role:
                base = c.split("_DAS")[0]
                if "_DAS" in c and base in BANDS:
                    role, unit, note = "spectral_band", "reflectance", "Interpolated to a common DAS"
                elif "_DAS" in c and base in index_names:
                    role, unit, note = "spectral_index", "index units", "Interpolated to a common DAS"
                elif "_DAS" in c and base in ROLE_UNITS:
                    # e.g. gsw_DAS80: a measured variable at a fixed date, not interpolated
                    brole, bunit, _ = ROLE_UNITS[base]
                    role, unit, note = brole, bunit, "%s at %s" % (base, c.split("_")[-1])
                elif c in BANDS:
                    role, unit = "spectral_band", "reflectance"
                elif c in index_names:
                    role, unit = "spectral_index", "index units"
                else:
                    # phenology-wide columns are "<Index>_<parameter>"; match the LONGEST
                    # parameter suffix, otherwise e.g. NDVI_AUC_pheno_SOS_EOS matches "EOS"
                    hits = [k for k in ROLE_UNITS if c.endswith("_" + k)]
                    if hits:
                        k = max(hits, key=len)
                        v = ROLE_UNITS[k]
                        role, unit, note = v[0], v[1], "%s for index %s" % (k, c[:-len(k) - 1])
                    else:
                        role, unit, note = "unclassified", "", "Not described in ROLE_UNITS"
            rows.append((tname, c, role, unit, note))
    return pd.DataFrame(rows, columns=["First_appears_in", "Column", "Role", "Unit", "Note"])


def build_coverage(tables):
    rows = []
    for tname, df in tables.items():
        if "Season" not in df.columns:
            continue
        for c in df.columns:
            if c in ("Season", "Season_ID"):
                continue
            g = df.groupby("Season")[c].apply(lambda s: int(s.notna().sum()))
            n = df.groupby("Season").size()
            rows.append({
                "Table": tname, "Column": c,
                "n_2024-25": int(g.get(S1, 0)), "of_2024-25": int(n.get(S1, 0)),
                "n_2025-26": int(g.get(S2, 0)), "of_2025-26": int(n.get(S2, 0)),
                "In_both_seasons": bool(g.get(S1, 0) > 0 and g.get(S2, 0) > 0),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. run
# ---------------------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(XLSX), exist_ok=True)
    os.makedirs(CSVD, exist_ok=True)

    print("harmonising design / harvest / physiology / ground ...")
    design = build_design()
    harvest, phys25 = build_harvest()
    phys = build_physiology(phys25)
    ground = build_ground()
    gsum = summarise_ground(ground)

    print("harmonising spectral series ...")
    spectral, vis = build_spectral()
    spec_interp = interp_common(spectral, vis)

    print("harmonising phenology and recomputing derived stages ...")
    pheno = build_phenology()
    phw = widen_phenology(pheno)

    print("assembling ML tables ...")
    ml_agro = build_ml_agronomic(design, harvest, phys, gsum)
    ml_phen = build_ml_phenology(phw, harvest, design)
    ml_agro_cs = drop_season_specific(ml_agro, label="ML_Agronomic_CrossSeason")
    ml_phen_cs = drop_season_specific(ml_phen, label="ML_Phenology_CrossSeason")

    tables = {
        "Plot_Design": design,
        "Harvest_PlotLevel": harvest,
        "Physiology_PlotLevel": phys,
        "Ground_TimeSeries": ground,
        "Ground_PlotSummary": gsum,
        "Spectral_TimeSeries": spectral,
        "Spectral_Interp_CommonDAS": spec_interp,
        "Phenology_Parameters": pheno,
        "Phenology_Wide": phw,
        "ML_Agronomic_vs_Yield": ml_agro,
        "ML_Phenology_vs_Yield": ml_phen,
        "ML_Agronomic_CrossSeason": ml_agro_cs,
        "ML_Phenology_CrossSeason": ml_phen_cs,
    }

    readme = build_readme()
    issues = build_known_issues()
    coverage = build_coverage(tables)
    dictionary = build_dictionary(tables)
    screening = rd(F24, "Index_Screening")

    print("writing workbook ...")
    with pd.ExcelWriter(XLSX, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        for name, df in tables.items():
            df.to_excel(xw, sheet_name=name, index=False)
        coverage.to_excel(xw, sheet_name="Coverage_Matrix", index=False)
        dictionary.to_excel(xw, sheet_name="Data_Dictionary", index=False)
        issues.to_excel(xw, sheet_name="Known_Issues", index=False)
        screening.to_excel(xw, sheet_name="Index_Screening_2024", index=False)

    print("writing CSVs ...")
    for name, df in tables.items():
        df.to_csv(os.path.join(CSVD, name + ".csv"), index=False)
    dictionary.to_csv(os.path.join(CSVD, "Data_Dictionary.csv"), index=False)
    coverage.to_csv(os.path.join(CSVD, "Coverage_Matrix.csv"), index=False)
    issues.to_csv(os.path.join(CSVD, "Known_Issues.csv"), index=False)

    print("\nDONE ->", XLSX)
    for name, df in tables.items():
        print("  %-28s %5d rows x %4d cols" % (name, len(df), df.shape[1]))
    return tables


if __name__ == "__main__":
    main()
