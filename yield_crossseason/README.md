# Yield prediction across seasons

Does UAV phenology predict wheat grain yield, and does a model trained in one season work in the next? The pipeline exists twice: once in **Python** (`python/`, canonical) and once in **R** (`R/`, an independent re-implementation used as a check). Both start from the two source workbooks and rebuild everything.

## Design of the analysis
- **Predictor blocks** kept separate so they can be compared: raw spectral trajectories, double-logistic parameters, agronomic traits, leaf physiology, ground time series, and their combinations. Circular predictors (biological yield → grain yield; either → harvest index) are blocked in code.
- **12 algorithms:** mean baseline, OLS, ridge, lasso, elastic net, PLSR, SVR (RBF), Gaussian process, random forest, extra trees, histogram gradient boosting, k-NN.
- **Nested cross-validation:** hyper-parameters tuned by an inner 4-fold CV inside every outer training fold.
- **Five validation schemes:** repeated 5-fold, leave-one-replication-out, pooled, raw between-season transfer, and between-season transfer after within-season standardisation. 6,444 configurations in total.
- **Statistics:** split-plot ANOVA with correct error strata; Benjamini–Hochberg FDR within season; associations called consistent only if they keep the same sign and survive correction in both seasons.
- **Optimism check:** replicates the common AIC-selected multiple regression workflow, scored in-sample and then inside cross-validation.

## Results (real data)
| | 2024–25 | 2025–26 |
|---|---|---|
| Raw spectral trajectories (best model R²) | 0.822 | 0.806 |
| Double-logistic parameters | 0.709 | 0.791 |
| Destructive agronomic traits | 0.724 | 0.717 |
| Experimental design only | 0.39 | 0.20 |
| Phenology, raw transfer to the other season | −0.03 | −0.25 |
| Phenology, standardised transfer | 0.76 | 0.58 |

- A single index value at 80 DAS correlated with yield at r ≈ 0.85 in both seasons. The correlation plateaus from about 60 DAS.
- Combining agronomic and phenological predictors added only 0.01–0.03 R² within season and cost 0.34–0.49 R² under transfer.
- AIC-selected regression overstated R² by 0.155 on average (maximum 0.383) compared with cross-validated estimates.
- With the correct split-plot strata, nitrogen changed canopy nitrogen status in one season but never grain yield; seaweed extract reached a single trait. An unstructured ANOVA inflates the count of responsive parameters about six-fold.

## Run (Python)
```bash
cd python
pip install -r requirements.txt
python wheat_paths.py          # check the resolved paths first
python run_all.py --skip-ml    # dataset + statistics + figures (about 2 min)
python run_all.py              # everything, incl. both ML sweeps (about 4.5 h on 12 cores)
```
Set `WHEAT_S1_XLSX` / `WHEAT_S2_XLSX` (or `WHEAT_ROOT`) to the source workbooks; see [../data/README.md](../data/README.md). Outputs go to `02_Dataset/`, `03_CSV/` and `04_ML/` beside this README (git-ignored).

| Script | What it does |
|---|---|
| `wheat_paths.py` | resolves every path; run it alone to check your setup |
| `build_combined_dataset.py` | harmonises both seasons, recomputes phenology, writes the tables |
| `verify_yield_discrepancy.py` | regression test for a 9-plot yield conflict in one source sheet |
| `univariate_analysis.py` | 519 parameters × 3 targets, BH correction, Type II ANOVA |
| `anova_splitplot.py` | split-plot tests with correct error strata |
| `figures_phenology.py`, `figures_descriptive.py` | phenology and descriptive figures (incl. *when to fly*) |
| `run_ml_analysis.py`, `run_ml_extended.py`, `run_ml_hybrid.py` | nested-CV model sweeps |
| `stepwise_vs_cv.py` | optimism of AIC-selected regression |
| `report_ml_analysis.py`, `model_tables.py`, `figures_extended.py`, `figures_manuscript.py` | summary tables and figures |

## Run (R)
```bash
cd R
Rscript install_packages.R
WHEAT_SKIP_ML=1 Rscript run_all.R     # dataset + statistics + figures (about 30 s)
Rscript run_all.R                     # add the ML sweep
```
Statistics and figures need only base R. R writes to `04_ML/results_R/` and `04_ML/figures_R/`, so it never overwrites the Python output.

**Python vs R:** the deterministic tables agree to floating-point precision (largest difference 1.8×10⁻⁵ days in one peak-of-season date). Machine-learning results agree within fold-to-fold noise. Fold assignment and tuning internals differ, so they cannot match exactly.
