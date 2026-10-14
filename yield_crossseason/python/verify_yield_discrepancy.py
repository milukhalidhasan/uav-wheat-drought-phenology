"""
Resolve the grain-yield discrepancy in MegaData2025_Cleaned.xlsx.

The 2025-26 Phenology_Parameters sheet carries a Grain_Yield(t/ha) column that
disagrees with that file's own Harvest_PlotLevel sheet for 9 of 96 plots.

Decisive test
-------------
Harvest index is defined as HI = grain yield / biological yield, and all three
columns sit in the harvest sheet. Whichever grain-yield value satisfies
HI x biological yield is the internally consistent one. On the 87 undisputed
plots this identity holds to a median of 0.003 t/ha, which sets the tolerance.

Result: for all 9 disputed plots the HARVEST sheet value reproduces HI x BY to
within 0.009 t/ha, while the phenology sheet value is off by exactly 1.00 t/ha
(eight plots) or 3.00 t/ha (one plot). The harvest sheet is correct; the
phenology sheet's yield column is stale. The combined dataset takes grain yield
from the harvest sheet and never propagates the phenology-sheet column, so no
downstream result depends on the erroneous values.

This script re-checks all of that and fails loudly if anything regresses.
"""
import os
import sys
import numpy as np
import pandas as pd

from wheat_paths import (ROOT, F24, F25, OUT, CSVD, XLSX, RESD, FIGD,
                         ML_DIR, REPORT_DOCX, check_inputs)

TOL = 0.02          # t/ha; generous next to the 0.003 median rounding error


def main():
    h = pd.read_excel(F25, sheet_name="Harvest_PlotLevel").set_index("Plot_ID")
    ph = (pd.read_excel(F25, sheet_name="Phenology_Parameters")
          .groupby("Plot_ID")["Grain_Yield(t/ha)"].first())

    d = pd.DataFrame({
        "GY_harvest": h["Grain_Yield(t/ha)"],
        "GY_phenology_sheet": ph,
        "Biological_Yield": h["Biological_Yield(t/ha)"],
        "Harvest_Index": h["Harvest_Index(HI)"]})
    d["GY_implied_by_HIxBY"] = d.Harvest_Index * d.Biological_Yield
    d["err_harvest"] = (d.GY_harvest - d.GY_implied_by_HIxBY).abs()
    d["err_phenology"] = (d.GY_phenology_sheet - d.GY_implied_by_HIxBY).abs()
    d["disputed"] = (d.GY_harvest - d.GY_phenology_sheet).abs() > 1e-9

    disputed = d[d.disputed]
    clean = d[~d.disputed]

    print("disputed plots: %d of %d" % (len(disputed), len(d)))
    print("\nidentity check  GY == Harvest_Index x Biological_Yield")
    print("  undisputed plots, median |GY_harvest - HI x BY| : %.4f t/ha"
          % clean.err_harvest.median())
    print("  disputed   plots, mean   |GY_harvest - HI x BY| : %.4f t/ha"
          % disputed.err_harvest.mean())
    print("  disputed   plots, mean   |GY_pheno   - HI x BY| : %.4f t/ha"
          % disputed.err_phenology.mean())
    print("\nper disputed plot")
    print(disputed[["GY_harvest", "GY_phenology_sheet", "Biological_Yield",
                    "Harvest_Index", "GY_implied_by_HIxBY", "err_harvest",
                    "err_phenology"]].round(4).to_string())

    ok = True
    if not (disputed.err_harvest < TOL).all():
        print("\nFAIL: a harvest-sheet value does not satisfy HI x BY")
        ok = False
    if not (disputed.err_phenology > TOL).all():
        print("\nFAIL: a phenology-sheet value unexpectedly satisfies HI x BY")
        ok = False

    # the combined dataset must carry the harvest-sheet values everywhere
    bad = disputed.index.tolist()
    print("\ncombined dataset uses the harvest-sheet values:")
    for f in ["Harvest_PlotLevel", "ML_Agronomic_vs_Yield", "ML_Phenology_vs_Yield",
              "ML_Agronomic_CrossSeason", "ML_Phenology_CrossSeason"]:
        path = os.path.join(CSVD, f + ".csv")
        if not os.path.exists(path):
            continue
        t = pd.read_csv(path)
        t = t[t.Season == "2025-26"].set_index("Plot_ID")
        diff = (t.loc[bad, "Grain_Yield_t_ha"] - d.loc[bad, "GY_harvest"]).abs().max()
        status = "OK" if diff < 1e-9 else "MISMATCH"
        print("  %-28s %s (max diff %.4f)" % (f, status, diff))
        if diff >= 1e-9:
            ok = False

    print("\nCONCLUSION: the harvest sheet is correct and is what the analysis uses."
          if ok else "\nCONCLUSION: verification FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
