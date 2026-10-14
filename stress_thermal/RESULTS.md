# Results: drought classification across seasons, and UAV thermal stress sensing

*Wheat split-plot trial, Gazipur Agricultural University: 2 water regimes × 4 N × 3 seaweed, 4 replications, 96 plots per season, seasons 2024–25 and 2025–26. Numbers below are from the real data; the demo data in this repository will not reproduce them.*

**Reading rule:** headline numbers are the **mean over all five classifiers** (LDA, logistic, SVM, random forest, gradient boosting). Picking the best model after seeing its score is optimistic. Best-model figures are written to `results/A2_classifier_summary_best_and_mean.csv` when the code is run.

---

## A. Can UAV canopy data tell drought from irrigated plots, and does that carry to a new season?

### A1. Within a season it is easy; across seasons it depends on the features (Fig 1)

| Feature set | Within season (leave-one-rep-out) | Other season, raw | Other season, standardised | Other season, 12-plot reference |
|---|---|---|---|---|
| **Double-logistic parameters** (6 VIs × 9) | 0.91 | 0.56 | **0.84** (0.85 / 0.83; AUC 0.94 / 0.88) | **0.71** (0.78 / 0.64) |
| Raw trajectories, 6 VIs (20–80 DAS) | 0.91 | 0.64 | 0.61 (0.48 / 0.75) | 0.59 |
| Raw trajectories, all bands + VIs | 0.93 | 0.62 | 0.60 | 0.55 |
| Single flight, 80 DAS | 0.88 | 0.61 | 0.67 | 0.63 |

*Balanced accuracy; 0.5 = chance. Pairs are 2024-25 → 2025-26 / 2025-26 → 2024-25.*

- **Within a season** every feature set separates the regimes (0.88–0.93).
- **Raw transfer fails** for all feature sets (0.56–0.64). This matches the yield result in the manuscript: absolute calibration does not carry between seasons.
- **After within-season standardisation, only the double-logistic parameters transfer well and consistently** (0.84 mean, both directions above 0.82). Raw trajectories become unstable: one direction is at chance (0.48).
- With a **small well-watered reference block** (12 irrigated plots from one replication of the new season used for scaling), the double-logistic parameters still reach 0.71 (AUC 0.88), ahead of every raw set.
- **Robustness** (`A6_*`): standardised transfer of the double-logistic parameters holds when drought plots are a minority of the new season. Balanced accuracy is 0.81–0.85 and AUC 0.90–0.93 at 20%, 33% and 50% drought share (30 random draws each).

**This reverses the yield result.** For *yield prediction* the raw trajectories beat the curve parameters (R² 0.82 vs 0.71). For *stress classification across seasons* the curve parameters beat the raw trajectories (0.84 vs 0.61). Curve parameters are the more **transferable stress descriptor**, even though they are not the best yield predictor.

### A2. What carries the signal: senescence timing (Fig 2)
- Permutation importance of the transferred classifier: **mid-senescence (t2) = 57% and 82%** of positive importance in the two directions. Green duration and senescence rate add most of the rest.
- Biology: **drought moved mid-senescence 4.3–11.7 days earlier, in all 6 indices, in both seasons, and in all 4 replications in every case** (`A7_senescence_shift_by_regime.csv`). Drought also shortened green duration and reduced the area under the curve.
- Contrast with yield: in the manuscript, area under the curve and peak carried 75–80% of yield importance. **Magnitude parameters predict yield; timing parameters identify stress.**

### A3. A single flight does not transfer (`A3_detection_by_date.csv`)
- Within season, one flight from about 50 DAS onward separates the regimes (balanced accuracy about 0.88–0.90, mean of two models).
- Across seasons, single flights stay near chance (0.49–0.58 raw; 0.33–0.62 standardised). Separating stress across seasons needs the **whole-season curve**, not one date. This contrasts with yield, where one flight at 70–80 DAS was enough.

---

## B. UAV thermal canopy temperature (2025–26 only; the 2024–25 sensor had no thermal band)

### B1. Hand-measured leaf physiology confirms the drought (split-plot tests, `B3_physiology_water_effect.csv`)
| | Irrigated | Drought | Split-plot test |
|---|---|---|---|
| Stomatal conductance gsw (mol m⁻² s⁻¹) | 0.043 | 0.028 (−35%) | F(1,3) = 25.2, p = 0.015 |
| Leaf temperature (°C) | 31.4 | 33.5 (+2.1) | F = 69.9, p = 0.004 |
| ΦPSII / ETR | 0.26 / 149 | 0.26 / 152 | n.s. |
| Grain yield (t ha⁻¹) | 4.92 | 3.64 | p = 0.019 |

Drought closed stomata and warmed leaves without damaging PSII.

### B2. UAV thermal tracks the drought (Fig 3)
- From **40 to 68 DAS, drought plots were 1.6–3.3 °C warmer** than irrigated plots. The difference was significant (BH q < 0.05, correct main-plot error) at 40, 47, 61 and 68 DAS, with D warmer in all 4 replications on those dates.
- No difference before drought set in (19–34 DAS) or at the end (78–83 DAS).
- Season means: canopy temperature +1.15 °C (p = 0.010). **Empirical CWSI 0.54 vs 0.36** (p = 0.006). Nitrogen and seaweed had no effect on thermal (p > 0.15).

### B3. Thermal relates to physiology and yield (Fig 5, Fig 6)
- Season-mean empirical CWSI vs **grain yield: r = −0.69** across all plots and **−0.61 within each regime**. The within-regime value shows it is not just the drought/irrigated contrast.
- CWSI vs **gsw: r = −0.32** (all plots), −0.24 within regime (p = 0.02). CWSI vs leaf temperature: r = 0.40 (all plots), not significant within regime.
- Per date, thermal correlates with yield at r ≈ −0.63 to −0.67 between 40 and 61 DAS. NDVI correlates at r ≈ +0.66 to +0.83 over the same window and peaks at +0.85 at 68 DAS.

### B4. Thermal for detection and prediction (Fig 4, `B5_*`, `B6_*`)
- **Thermal alone at 40 DAS:** balanced accuracy 0.91 (leave-one-rep-out), better than the multispectral data on the same flight (0.88). Thermal + multispectral reached 0.96. That is the earliest date with a clear drought signal in this season.
- Whole season: thermal only 0.98, multispectral 0.94, both 0.96.
- **Yield prediction:** thermal series alone R² 0.44–0.60. Adding thermal to the 6-VI series raised R² from 0.79 to 0.82 (ridge, leave-one-rep-out). That gain is **within fold-to-fold noise at n = 96**, so it is not claimed as real.

---

## C. Supplementary: UAV canopy height vs ruler height (FigS1)
- Per-date r = 0.35–0.47 from 40 DAS (pooled r = 0.46). Photogrammetric canopy height underestimates and is noisy for wheat at this resolution.
- Treated as a weak result; LiDAR would be the natural next step for canopy height.

---

## Caveats (keep these honest in any write-up)
1. **Split-plot layout.** Water regime sits on main plots. Within-season classification could partly exploit field position of the main plots. Leave-one-rep-out and cross-season tests reduce this, but cannot rule it out if the layout was similar both seasons.
2. **Standardised transfer needs a batch of plots from the new season** and assumes both classes are present. This was tested down to a 20% drought share; the reference-block scheme avoids the assumption at a cost in accuracy.
3. **Thermal is one season.** Plot-mean temperature mixes canopy and soil (no canopy masking), so early-season thermal–yield correlations (19–29 DAS, before any drought difference) probably reflect ground cover. There is no air temperature or VPD, so the CWSI is an empirical within-date index, not the physical CWSI. 34 DAS was a cold flight. The 68 DAS sign flip in thermal–yield correlation is unexplained.
4. **2025–26 leaf physiology** is a single measurement with no recorded date.
5. **One site, one variety, two seasons.** These are *treatment-level* stress signatures. Genotypic variation and heritability cannot be tested with this trial. That is exactly what a PhD in a phenotyping lab would add.

---
