"""Shared paths, loaders and feature builders for the drought-classifier and thermal analyses.

Inputs are the harmonised two-season tables written by yield_crossseason/python/build_combined_dataset.py
(see data/README.md).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
# Input tables: $WHEAT_CSV_DIR if set, else data/full (real data, git-ignored), else data/demo (synthetic)
DATA = Path(os.environ.get("WHEAT_CSV_DIR") or
            (REPO / "data" / "full" if (REPO / "data" / "full" / "Plot_Design.csv").exists()
             else REPO / "data" / "demo"))
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SEASONS = ["2024-25", "2025-26"]
KEY = ["Season", "Plot_UID", "Water_Regime", "Replication", "Treatment"]

# Six indices carried through the double-logistic analysis (column prefixes in Phenology_Wide)
DL_INDICES = ["CIrededge", "GCI", "NDRE", "NDVI", "RVI", "gNDVI"]
RAW6 = ["CI(rededge)", "GCI", "NDRE", "NDVI", "RVI", "gNDVI"]
# Curve parameters that are comparable across seasons. SOS/EOS/LOS and the native AUC are
# excluded because they are often extrapolated beyond the flown window (see Known_Issues.csv).
DL_PARAMS = ["Baseline", "Amplitude", "Peak_value", "t1_mid_greenup", "t2_mid_senescence",
             "k1_rate_greenup", "k2_rate_senescence", "AUC_common_19_83", "Green_Duration"]
# Exact duplicates of other indices (Known_Issues.csv)
DUPLICATE_INDICES = ["TCARI", "TCARI/OSAVI"]
COMMON_DAS = [20, 30, 40, 50, 60, 70, 80]


def label(df):
    """1 = managed drought, 0 = irrigated."""
    return (df["Water_Regime"] == "D").astype(int).to_numpy()


def load_dl():
    w = pd.read_csv(DATA / "Phenology_Wide.csv")
    cols = [f"{i}_{p}" for i in DL_INDICES for p in DL_PARAMS]
    return w[KEY + cols].copy(), cols


def load_interp():
    s = pd.read_csv(DATA / "Spectral_Interp_CommonDAS.csv")
    feat = [c for c in s.columns[7:] if c.rsplit("_DAS", 1)[0] not in DUPLICATE_INDICES]
    return s[KEY + feat].copy(), feat


def interp_cols(feat, variables=None, das=None):
    out = []
    for c in feat:
        var, d = c.rsplit("_DAS", 1)
        if variables is not None and var not in variables:
            continue
        if das is not None and int(d) not in das:
            continue
        out.append(c)
    return out


def load_feature_sets():
    """Return {name: (frame, columns)} for the cross-season classifier."""
    dl, dl_cols = load_dl()
    sp, sp_cols = load_interp()
    return {
        "Double-logistic parameters (6 VIs x 9)": (dl, dl_cols),
        "Raw trajectories, 6 VIs (20-80 DAS)": (sp, interp_cols(sp_cols, RAW6)),
        "Raw trajectories, all bands + VIs (20-80 DAS)": (sp, sp_cols),
        "Single date, 80 DAS (all bands + VIs)": (sp, interp_cols(sp_cols, das=[80])),
    }


def load_2526_native():
    """2025-26 spectral time series on its native flight dates (thermal and dCHM present)."""
    ts = pd.read_csv(DATA / "Spectral_TimeSeries.csv")
    ts = ts[ts["Season"] == "2025-26"].copy()
    drop = ["Season_ID", "Plot_ID", "Flight_Date", "DOY_calendar", "Flight_No", "Growth_Stage", "SR"]
    ts = ts.drop(columns=[c for c in drop + DUPLICATE_INDICES if c in ts.columns])
    return ts


def zscore_within(df, cols, by="Season", ref_mask=None):
    """Standardise columns within each group. If ref_mask is given, the mean and SD come
    only from the rows where ref_mask is True (e.g. a well-watered reference block)."""
    out = df.copy()
    for _, idx in df.groupby(by).groups.items():
        block = df.loc[idx, cols]
        ref = block if ref_mask is None else block[ref_mask.loc[idx]]
        mu, sd = ref.mean(), ref.std(ddof=0).replace(0, np.nan)
        out.loc[idx, cols] = ((block - mu) / sd).fillna(0.0).to_numpy()
    return out
