# Data

## What is here
| Path | Content |
|---|---|
| `Data_Dictionary.csv` | every column of every table: table, column, variable family, unit, notes |
| `demo/` | **synthetic** versions of the harmonised tables, for running the code. Not real measurements; see `demo/README_SYNTHETIC.txt` |
| `full/` | *(not in the repository; git-ignored)*: put the real harmonised tables here |

## Availability of the real data
The field measurements (two seasons, 192 plot-season records: UAV spectral and thermal time series, double-logistic parameters, leaf physiology, harvest data) are unpublished. They will be released with the paper. Until then, please contact the author.

## Where the tables come from
`yield_crossseason/python/build_combined_dataset.py` (or `R/01_build_combined_dataset.R`) reads the two source workbooks:

| Season | Workbook | Flights |
|---|---|---|
| 2024–25 | `Wheat_2024-25_Structured_Dataset.xlsx` | 11 (10–108 DAS), multispectral |
| 2025–26 | `MegaData2025_Cleaned.xlsx` | 10 (19–83 DAS), multispectral + thermal |

It then harmonises them. The two seasons used different but equivalent double-logistic parameterisations, which are mapped to one form. Stage metrics are recomputed. Indices are interpolated to 20, 30 … 80 DAS. Areas under the curve are re-integrated over the shared 19–83 DAS window. Point the build step at the workbooks with:

```bash
export WHEAT_S1_XLSX=/path/to/Wheat_2024-25_Structured_Dataset.xlsx
export WHEAT_S2_XLSX=/path/to/MegaData2025_Cleaned.xlsx
python yield_crossseason/python/build_combined_dataset.py
```

The built tables land in `yield_crossseason/03_CSV/`. For the stress analyses, copy them to `data/full/`, or set `WHEAT_CSV_DIR` to that folder.

## Things to know about the data
- **Flight windows differ** between seasons. Use `AUC_common_19_83` and the interpolated 20–80 DAS columns for anything that pools seasons.
- **Thermal and per-flight canopy height exist only in 2025–26.**
- **Leaf physiology:** the mean of 80 and 88 DAS in 2024–25; a single undated measurement in 2025–26. Compare only as season-level summaries.
- **Season offset:** similar yields came from very different canopies (harvest index 0.43 vs 0.36). Centre within season or model season explicitly.
- **SOS and EOS are often extrapolated** beyond the flown window. Check `n_stages_outside_native` before using them.
- **Split-plot error strata:** water regime is the main-plot factor. Test it against replication × water regime, not the subplot residual.
