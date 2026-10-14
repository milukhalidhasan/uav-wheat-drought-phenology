# Drought-stress phenotyping and UAV thermal

Two analyses on the same two-season trial. Findings and caveats from the real data are in [RESULTS.md](RESULTS.md).

1. **Drought vs irrigated classification, within and between seasons.** Four feature sets (double-logistic parameters, raw trajectories of 6 VIs, raw trajectories of all bands and VIs, a single flight at 80 DAS) and five classifiers (shrinkage LDA, L2 logistic, RBF SVM, random forest, gradient boosting). Follow-ups:
   - detection date by date;
   - permutation importance of the curve parameters;
   - robustness when drought plots are a minority of the new season;
   - the direction of the senescence shift.
2. **UAV thermal (2025–26):**
   - canopy temperature by water regime with split-plot tests;
   - two relative stress indices (temperature excess over irrigated plots; empirical CWSI with per-date statistical baselines);
   - agreement with hand-measured gsw, leaf temperature, ΦPSII and ETR;
   - correlation with yield by date;
   - whether thermal adds to multispectral data for detection and yield prediction.

   A supplementary check compares UAV canopy height with ruler height.

## Validation schemes
| Code | Meaning |
|---|---|
| `LORO` | within one season, leave one replication out (12 drought + 12 irrigated plots per fold) |
| `Tr` | train on one season, test on the other, raw features |
| `TrZ` | as `Tr`, features z-scored within each season (label-free; needs a batch of plots from the new season) |
| `TrRef` | as `Tr`, features scaled by well-watered reference plots. In the new season only the 12 irrigated plots of one replication are used; scored on the other 84 plots and averaged over the 4 possible reference blocks |

Hyper-parameters are always tuned by an inner group-wise CV on the training data only. Headline numbers are the **mean over classifiers**, not the best one.

## Run
```bash
cd code
pip install -r requirements.txt
python run_all.py        # about 6 min on 12 cores
```
Input: `data/full/` if it exists, else `$WHEAT_CSV_DIR`, else the synthetic `data/demo/`. Outputs: `results/` (CSV tables A1–A7, B0–B6, C) and `figures/`, both git-ignored. Curated copies of the real-data figures are in [`../figures/`](../figures/).

| Script | Steps |
|---|---|
| `common.py` | paths, loaders, feature sets, within-season standardisation |
| `a_drought_classifier.py` | A1–A5: classifier sweep, detection by date, importance |
| `a2_robustness.py` | A6–A7: class-imbalance test, senescence shift |
| `b_thermal_analysis.py` | B0–B6, C: thermal, CWSI, physiology, yield, canopy height |
| `figures.py` | all figures (palette checked for colour-blind separation) |
