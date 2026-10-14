# Double-logistic curve fitting (per season)

One R script per season fits the double-logistic model to every vegetation index of every plot. The same scripts also run the per-season screening, split-plot tests and figures.

| Script | Season | Input (not in this repository) | Environment variable |
|---|---|---|---|
| `fit_double_logistic_2024-25.R` | 2024–25, 11 flights (10–108 DAS) | `Wheat_2024_Megadata_Latest.xlsx` | `WHEAT_S1_RAW_XLSX` |
| `fit_double_logistic_2025-26.R` | 2025–26, 10 flights (19–83 DAS) | `MegaData2025_Cleaned.xlsx` | `WHEAT_S2_XLSX` |

## The model
```
y(t) = y0 + (y1 - y0) * [ 1/(1+exp(-k1 (t - t1))) - 1/(1+exp(-k2 (t - t2))) ]
```
- `y0`: background value; `y1`: plateau value; `t1`, `t2`: mid-points of green-up and senescence; `k1`, `k2`: their rates.
- This is the lab's seven-parameter form `a + b/(1+exp(-c(t-d))) - e/(1+exp(-f(t-g)))` with `b = e`.
- `yield_crossseason/python/build_combined_dataset.py` maps both seasons onto one parameterisation (Amplitude = y1 − y0).

## How the fit works (`fit_generic`)
- Non-linear least squares with `minpack.lm::nlsLM` (Levenberg–Marquardt), separately for every plot × index.
- **Bounds scaled to each series:**
  - baseline and plateau are relative to the observed range;
  - `t1` and `t2` are relative to the flight window;
  - `k1` and `k2` lie in [0.01, 0.80] per day.
- **Outputs per fit:** R², RMSE, and `at_bound` (how many parameters ended on a bound), so weak fits can be flagged.
- **Group-level curves** (by water regime or treatment) are fitted afterwards, starting from the plot-level estimates.

## Run
```bash
Rscript -e 'install.packages(c("readxl","dplyr","tidyr","purrr","ggplot2","minpack.lm"))'
WHEAT_S2_XLSX=/path/to/MegaData2025_Cleaned.xlsx Rscript fit_double_logistic_2025-26.R
```
Outputs (parameter tables, tests, figures) are written to `outputs/` or `outputs_2024/` next to where the script is run.
