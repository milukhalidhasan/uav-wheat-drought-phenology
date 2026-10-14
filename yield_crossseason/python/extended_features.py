"""
Every available parameter, assembled into one plot-season matrix and grouped into
predictor blocks for the extended analysis.

Blocks
------
Design            water regime, nitrogen, seaweed
Agro_predictive   non-component harvest traits + stress biochemistry
Agro_components   yield components (definitional, kept separate)
Physio            shared gas exchange / fluorescence (4 vars, both seasons)
Physio_extra24    full LI-COR panel, 2024-25 only, incl. per-date values
Ground_summary    campaign summaries of chlorophyll / height / soil
Ground_ts         time-resolved ground data (per measurement date, within season)
Pheno             double-logistic curve parameters, 6 indices
Spectral          every band and index interpolated to 20-80 DAS
Structure24       dCHM canopy height, 2024-25 only
"""
import os
import re
import numpy as np
import pandas as pd

from wheat_paths import (ROOT, F24, F25, OUT, CSVD, XLSX, RESD, FIGD,
                         ML_DIR, REPORT_DOCX, check_inputs)

S1, S2 = "2024-25", "2025-26"
KEYS = ["Season", "Season_ID", "Plot_UID", "Plot_ID", "Water_Regime", "Treatment",
        "N_Level", "N_kg_ha", "Seaweed_Level", "Seaweed_pct", "Replication"]
TARGETS = ["Grain_Yield_t_ha", "Biological_Yield_t_ha", "Harvest_Index"]
DESIGN_CAT = ["Water_Regime", "N_Level", "Seaweed_Level"]

COMPONENTS = ["Tiller_No", "Spike_Length_cm", "Total_Grain_per_Plant", "TSW_g",
              "Fresh_Biomass_t_ha"]
FORBIDDEN = {
    "Grain_Yield_t_ha": ["Biological_Yield_t_ha", "Harvest_Index"],
    "Biological_Yield_t_ha": ["Grain_Yield_t_ha", "Harvest_Index"],
    "Harvest_Index": ["Grain_Yield_t_ha", "Biological_Yield_t_ha"],
}

rd = lambda n: pd.read_csv(os.path.join(CSVD, n + ".csv"))


def ground_timeseries_wide():
    """One column per ground variable per measurement date. The two seasons were
    sampled on different dates, so these columns are season-specific by nature."""
    g = rd("Ground_TimeSeries")
    vars_ = ["Chlorophyll", "Plant_height_cm", "Soil_Temp_C", "Soil_Moisture_pct"]
    w = g.pivot_table(index="Plot_UID", columns="DAS", values=vars_)
    w.columns = ["%s_D%d" % (v, d) for v, d in w.columns]
    w = w.dropna(axis=1, how="all")
    return w.reset_index()


def build():
    design = rd("Plot_Design")[["Season", "Season_ID", "Plot_UID", "Plot_ID", "Water_Regime",
                                "Treatment", "N_Level", "N_kg_ha", "Seaweed_Level",
                                "Seaweed_pct", "Replication", "dCHM_relative_plot"]]
    harvest = rd("Harvest_PlotLevel").drop(columns=["Season", "Season_ID", "Plot_ID",
                                                    "Water_Regime", "Treatment", "Replication"])
    physio = rd("Physiology_PlotLevel").drop(columns=["Season", "Season_ID", "Plot_ID",
                                                      "Physiology_Source"])
    gsum = rd("Ground_PlotSummary").drop(columns=["Season", "Season_ID", "Plot_ID"])
    gts = ground_timeseries_wide()
    pheno = rd("ML_Phenology_CrossSeason").drop(
        columns=[c for c in KEYS if c != "Plot_UID"] + TARGETS, errors="ignore")
    spec = rd("Spectral_Interp_CommonDAS").drop(
        columns=["Season", "Season_ID", "Plot_ID", "Water_Regime", "Replication", "Treatment"])

    df = design
    for part in (harvest, physio, gsum, gts, pheno, spec):
        df = df.merge(part, on="Plot_UID", how="left")
    assert len(df) == 192, len(df)

    agro_all = ["Plant_Length_cm", "Root_Length_cm", "Fresh_Biomass_t_ha", "Tiller_No",
                "Spike_Length_cm", "Total_Grain_per_Plant", "TSW_g",
                "H2O2_umol_per_gFW", "MDA_nmol_per_gFW"]
    blocks = {
        "Agro_predictive": [c for c in agro_all if c not in COMPONENTS and c != "MDA_nmol_per_gFW"],
        "Agro_components": COMPONENTS,
        "Physio": ["gsw", "PhiPS2", "ETR", "Tleaf"],
        "Physio_extra24": ["gbw", "gtw", "E_apparent", "VPleaf", "VPDleaf", "H2O_leaf",
                           "Fs", "Fm.", "rh_s", "Qamb", "gsw_DAS80", "gsw_DAS88",
                           "PhiPS2_DAS80", "PhiPS2_DAS88", "ETR_DAS80", "ETR_DAS88",
                           "Tleaf_DAS80", "Tleaf_DAS88"],
        "Ground_summary": ["Chlorophyll_mean", "Chlorophyll_max", "Chlorophyll_last",
                           "Plant_height_max_cm", "Plant_height_mean_cm",
                           "Soil_Temp_mean_C", "Soil_Moisture_mean_pct"],
        "Ground_ts": [c for c in gts.columns if c != "Plot_UID"],
        "Stress": ["MDA_nmol_per_gFW"],
        "Structure24": ["dCHM_relative_plot"],
        "Pheno": [c for c in pheno.columns if c != "Plot_UID" and not c.endswith("_R2")],
        "Spectral": [c for c in spec.columns if c != "Plot_UID"],
    }
    blocks = {k: [c for c in v if c in df.columns] for k, v in blocks.items()}
    return df, blocks


def cols_present(df, cols, season=None):
    """Keep only columns with at least two distinct non-null values in the data used."""
    d = df if season is None else df[df.Season == season]
    return [c for c in cols if c in d.columns and d[c].notna().sum() > 2
            and d[c].nunique(dropna=True) > 1]


def feature_sets(df, blocks):
    """Predictor sets for the extended sweep. Sets marked *_24 are 2024-25 only."""
    B = blocks
    both = lambda *ks: sum([B[k] for k in ks], [])
    return {
        # cross-season capable
        "Design_only":       ([], DESIGN_CAT, "both"),
        "Agro_predictive":   (both("Agro_predictive", "Physio", "Ground_summary"), [], "both"),
        "Agro_full":         (both("Agro_predictive", "Agro_components", "Physio",
                                   "Ground_summary"), [], "both"),
        "Physio_only":       (B["Physio"], [], "both"),
        "Ground_summary":    (B["Ground_summary"], [], "both"),
        "Pheno":             (B["Pheno"], [], "both"),
        "Spectral":          (B["Spectral"], [], "both"),
        # everything a drone can deliver: curve parameters + raw index trajectories
        "All_remote":        (both("Pheno", "Spectral"), DESIGN_CAT, "both"),
        "All_parameters":    (both("Agro_predictive", "Physio", "Ground_summary",
                                   "Pheno", "Spectral"), DESIGN_CAT, "both"),
        # within-season only (season-specific measurements)
        "Ground_ts":         (B["Ground_ts"], [], "within"),
        "Physio_extra24":    (both("Physio", "Physio_extra24"), [], "s1"),
        "Extras24":          (both("Stress", "Structure24"), [], "s1"),
    }
