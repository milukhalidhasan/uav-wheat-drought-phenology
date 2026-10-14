# ---------------------------------------------------------------------------
# Publication figures, base R graphics (no ggplot2 needed).
# R port of 01_Code/figures_phenology.py + figures_descriptive.py.
#
# Dependencies: BASE R ONLY.
# Run: Rscript 04_figures.R          (after 01 and 02)
#
# Writes PNGs to 04_ML/figures_R/:
#   R_P1_curves_by_water          fitted double-logistic curves, 6 indices x 2 seasons
#   R_P2_curves_by_nitrogen       fitted NDVI curves by nitrogen level
#   R_P3_parameter_distributions  curve parameters by season x water regime
#   R_P4_stages                   derived phenological stages on a DAS axis
#   R_P5_fit_quality              curve fit R2 / RMSE / parameters at bound
#   R_P6_auc_vs_yield             AUC vs grain yield per index
#   R_P7_observed_trajectories    observed index means over time
#   R_P8_ground_timeseries        ground measurements over time
#   R_D1_treatment_response       yield and components by N rate and water regime
#   R_D2_when_to_fly              index-to-yield correlation vs flight date
#   R_D3_top_univariate           strongest associations present in both seasons
#   R_D4_correlation_matrix       correlation structure among headline parameters
#   R_D5_season_shift             Hedges' g between seasons
#   R_D6_treatment_effects        share of parameters responding to each factor
# ---------------------------------------------------------------------------
source("00_config.R")

SUF <- if (file.exists(file.path(CSVD, "Plot_Design_R.csv"))) "_R" else ""
rdc <- function(n) read.csv(file.path(CSVD, paste0(n, SUF, ".csv")),
                            check.names = FALSE, stringsAsFactors = FALSE)
rdr <- function(n) {
  f <- file.path(RESD, paste0(n, ".csv"))
  if (!file.exists(f)) { warning("missing ", f, " - run 02_univariate_analysis.R"); return(NULL) }
  read.csv(f, check.names = FALSE, stringsAsFactors = FALSE)
}

PAR <- function(...) par(mar = c(4, 4.2, 2.6, 1), mgp = c(2.4, 0.7, 0), tcl = -0.3,
                         cex.axis = 0.85, cex.lab = 0.95, las = 1, bty = "n", ...)
grid0 <- function() grid(col = "grey88", lty = 1, lwd = 0.7)
se <- function(v) { v <- v[is.finite(v)]; if (length(v) < 2) NA_real_ else sd(v)/sqrt(length(v)) }

# ---------------------------------------------------------------------------
ph     <- rdc("Phenology_Parameters")
design <- rdc("Plot_Design")
ph <- merge(ph, design[, c("Plot_UID", "N_Level", "Seaweed_Level", "N_kg_ha")],
            by = "Plot_UID")
harvest <- rdc("Harvest_PlotLevel")
spec    <- rdc("Spectral_TimeSeries")
ground  <- rdc("Ground_TimeSeries")
specint <- rdc("Spectral_Interp_CommonDAS")
agroX   <- rdc("ML_Agronomic_CrossSeason")

mean_curve <- function(sub, t) {
  if (!nrow(sub)) return(list(m = rep(NA, length(t)), se = rep(NA, length(t))))
  Y <- vapply(seq_len(nrow(sub)), function(i)
    dlogistic(t, sub$Baseline[i], sub$Amplitude[i], sub$t1_mid_greenup[i],
              sub$t2_mid_senescence[i], sub$k1_rate_greenup[i], sub$k2_rate_senescence[i]),
    numeric(length(t)))
  list(m = rowMeans(Y), se = apply(Y, 1, sd) / sqrt(ncol(Y)))
}

band <- function(t, m, s, col) {
  ok <- is.finite(m) & is.finite(s)
  polygon(c(t[ok], rev(t[ok])), c((m - 1.96 * s)[ok], rev((m + 1.96 * s)[ok])),
          col = adjustcolor(col, 0.18), border = NA)
}

# --- P1: fitted curves by water regime -------------------------------------
open_png("R_P1_curves_by_water", 2600, 1100, 150)
par(mfrow = c(2, length(INDICES)), oma = c(0, 2.4, 3, 0)); PAR()
for (season in c(S1, S2)) {
  w <- FLIGHT_WINDOW[[season]]; t <- seq(w[1], w[2], length.out = 300)
  for (idx in INDICES) {
    cd <- mean_curve(ph[ph$Season == season & ph$Index == idx & ph$Water_Regime == "D", ], t)
    ci <- mean_curve(ph[ph$Season == season & ph$Index == idx & ph$Water_Regime == "I", ], t)
    yl <- range(c(cd$m, ci$m), na.rm = TRUE)
    plot(t, cd$m, type = "n", ylim = yl, xlab = "Days after sowing", ylab = "",
         main = if (season == S1) idx else "")
    grid0(); band(t, cd$m, cd$se, COL_D); band(t, ci$m, ci$se, COL_I)
    lines(t, cd$m, col = COL_D, lwd = 2.4); lines(t, ci$m, col = COL_I, lwd = 2.4)
    if (idx == INDICES[1]) {
      mtext(season, side = 2, line = 3.1, font = 2, cex = 0.8)
      if (season == S1) legend("bottomright", c("Drought", "Irrigated"),
                               col = c(COL_D, COL_I), lwd = 2.4, bty = "n", cex = 0.85)
    }
  }
}
mtext("Fitted double-logistic canopy trajectories by water regime (mean +/- 95% CI)",
      outer = TRUE, line = 0.8, font = 2, cex = 1.1)
dev.off()

# --- P2: fitted NDVI curves by nitrogen ------------------------------------
open_png("R_P2_curves_by_nitrogen", 1700, 1300, 150)
par(mfrow = c(2, 2), oma = c(0, 0, 3, 0)); PAR()
ncol4 <- c("#CBD5E0", "#90CDF4", "#4299E1", "#2A4365")
for (season in c(S1, S2)) {
  w <- FLIGHT_WINDOW[[season]]; t <- seq(w[1], w[2], length.out = 300)
  for (wr in c("D", "I")) {
    curves <- lapply(paste0("N", 0:3), function(nl)
      mean_curve(ph[ph$Season == season & ph$Index == "NDVI" &
                    ph$Water_Regime == wr & ph$N_Level == nl, ], t)$m)
    plot(t, curves[[1]], type = "n", ylim = range(unlist(curves), na.rm = TRUE),
         xlab = "Days after sowing", ylab = "NDVI",
         main = sprintf("%s - %s", season, ifelse(wr == "D", "Drought", "Irrigated")))
    grid0()
    for (k in 1:4) lines(t, curves[[k]], col = ncol4[k], lwd = 2.4)
    if (season == S1 && wr == "D")
      legend("bottomright", sprintf("N%d (%d kg/ha)", 0:3, c(0, 210, 300, 390)),
             col = ncol4, lwd = 2.4, bty = "n", cex = 0.8, title = "Nitrogen")
  }
}
mtext("Fitted NDVI trajectories by nitrogen level", outer = TRUE, line = 0.8,
      font = 2, cex = 1.1)
dev.off()

# --- P3: parameter distributions -------------------------------------------
pars <- c("Baseline","Amplitude","Peak_value","t1_mid_greenup","t2_mid_senescence",
          "k1_rate_greenup","k2_rate_senescence","Green_Duration","LOS",
          "AUC_common_19_83","Rate_greenup","Rate_senescence")
nd <- ph[ph$Index == "NDVI", ]
open_png("R_P3_parameter_distributions", 2200, 1500, 150)
par(mfrow = c(3, 4), oma = c(0, 0, 3, 0)); PAR()
for (p in pars) {
  grp <- list(); labs <- c(); cols <- c()
  for (season in c(S1, S2)) for (wr in c("D", "I")) {
    grp[[length(grp) + 1]] <- nd[[p]][nd$Season == season & nd$Water_Regime == wr]
    labs <- c(labs, paste0(substr(season, 6, 7), "\n", wr))
    cols <- c(cols, if (season == S1) COL_S1 else COL_S2)
  }
  boxplot(grp, names = labs, col = adjustcolor(cols, 0.75), outline = FALSE,
          main = p, border = "grey25", cex.main = 0.95)
  grid0()
}
mtext("Double-logistic parameter distributions (NDVI), season x water regime",
      outer = TRUE, line = 0.8, font = 2, cex = 1.1)
dev.off()

# --- P4: derived stages ----------------------------------------------------
stages <- c("SOS","EOG","POS","SOD","EOS")
open_png("R_P4_stages", 2000, 1100, 150)
par(mfrow = c(1, 2), oma = c(0, 0, 3, 0)); PAR(mar = c(4, 7, 2.6, 1))
for (season in c(S1, S2)) {
  rows <- expand.grid(wr = c("D", "I"), idx = INDICES, stringsAsFactors = FALSE)
  plot(NA, xlim = c(-30, 150), ylim = c(nrow(rows) + 0.5, 0.5), xlab = "Days after sowing",
       ylab = "", yaxt = "n", main = paste0(season, "   (shaded = flight window)"))
  w <- FLIGHT_WINDOW[[season]]
  rect(w[1], -1, w[2], nrow(rows) + 2, col = adjustcolor("#68D391", 0.12), border = NA)
  abline(v = w, col = "grey60", lty = 3); grid0()
  for (i in seq_len(nrow(rows))) {
    sub <- ph[ph$Season == season & ph$Index == rows$idx[i] &
              ph$Water_Regime == rows$wr[i], ]
    mu <- vapply(stages, function(s) mean(sub[[s]], na.rm = TRUE), numeric(1))
    sg <- vapply(stages, function(s) sd(sub[[s]], na.rm = TRUE), numeric(1))
    col <- if (rows$wr[i] == "D") COL_D else COL_I
    lines(mu, rep(i, length(mu)), col = col, lwd = 1.2)
    arrows(mu - sg, i, mu + sg, i, angle = 90, code = 3, length = 0.02, col = col, lwd = 0.8)
    points(mu, rep(i, length(mu)), pch = 19, col = col, cex = 0.7)
  }
  axis(2, at = seq_len(nrow(rows)), labels = paste(rows$idx, rows$wr), cex.axis = 0.65)
}
mtext("Derived phenological stages: SOS, EOG, POS, SOD, EOS (mean +/- SD)",
      outer = TRUE, line = 0.8, font = 2, cex = 1.1)
dev.off()

# --- P5: fit quality -------------------------------------------------------
open_png("R_P5_fit_quality", 2000, 750, 150)
par(mfrow = c(1, 3), oma = c(0, 0, 3, 0)); PAR()
agg <- function(f) sapply(c(S1, S2), function(s)
  sapply(INDICES, function(i) f(ph[ph$Season == s & ph$Index == i, ])))
for (spec3 in list(
      list(v = agg(function(d) mean(d$R2)), t = "Curve fit R2", yl = "R2", ylim = c(0.5, 1.02)),
      list(v = agg(function(d) mean(d$RMSE)), t = "Curve fit RMSE", yl = "RMSE", ylim = NULL),
      list(v = agg(function(d) 100 * mean(d$Params_At_Bound > 0)),
           t = "Parameters at a bound", yl = "% of plots", ylim = NULL))) {
  bp <- barplot(t(spec3$v), beside = TRUE, col = c(COL_S1, COL_S2), border = NA,
                names.arg = INDICES, las = 2, main = spec3$t, ylab = spec3$yl,
                ylim = spec3$ylim, cex.names = 0.75)
  grid0()
  if (spec3$t == "Curve fit R2") legend("bottomright", c(S1, S2), fill = c(COL_S1, COL_S2),
                                        border = NA, bty = "n", cex = 0.85)
}
mtext("Double-logistic fit quality by index and season", outer = TRUE, line = 0.8,
      font = 2, cex = 1.1)
dev.off()

# --- P6: AUC vs grain yield ------------------------------------------------
gy <- setNames(harvest$Grain_Yield_t_ha, harvest$Plot_UID)
open_png("R_P6_auc_vs_yield", 2600, 1100, 150)
par(mfrow = c(2, length(INDICES)), oma = c(0, 2.4, 3, 0)); PAR()
for (season in c(S1, S2)) for (idx in INDICES) {
  sub <- ph[ph$Season == season & ph$Index == idx, ]
  sub$GY <- unname(gy[sub$Plot_UID])
  plot(sub$AUC_common_19_83, sub$GY, type = "n",
       xlab = "AUC (19-83 DAS)", ylab = "", main = if (season == S1) idx else "")
  grid0()
  for (wr in c("D", "I")) {
    s <- sub[sub$Water_Regime == wr, ]
    points(s$AUC_common_19_83, s$GY, pch = 19, cex = 0.55,
           col = adjustcolor(if (wr == "D") COL_D else COL_I, 0.75))
  }
  ok <- is.finite(sub$AUC_common_19_83) & is.finite(sub$GY)
  if (sum(ok) > 3) {
    fit <- lm(GY ~ AUC_common_19_83, data = sub[ok, ]); abline(fit, lwd = 1.6)
    legend("topleft", sprintf("r = %.2f", cor(sub$AUC_common_19_83[ok], sub$GY[ok])),
           bty = "n", cex = 0.85, text.font = 2)
  }
  if (idx == INDICES[1]) mtext(paste(season, "\nGrain yield (t/ha)"), side = 2,
                               line = 2.6, cex = 0.7, font = 2)
}
mtext("Cross-season-comparable AUC vs grain yield, by index", outer = TRUE,
      line = 0.8, font = 2, cex = 1.1)
dev.off()

# --- P7: observed trajectories ---------------------------------------------
open_png("R_P7_observed_trajectories", 2600, 1100, 150)
par(mfrow = c(2, length(INDICES)), oma = c(0, 2.4, 3, 0)); PAR()
for (season in c(S1, S2)) for (idx in INDICES) {
  d <- spec[spec$Season == season, ]
  ds <- lapply(c("D", "I"), function(wr) {
    s <- d[d$Water_Regime == wr, ]
    data.frame(DAS = sort(unique(s$DAS)),
               m  = tapply(s[[idx]], s$DAS, mean, na.rm = TRUE),
               se = tapply(s[[idx]], s$DAS, se))
  })
  yl <- range(unlist(lapply(ds, function(x) c(x$m - 2*x$se, x$m + 2*x$se))), na.rm = TRUE)
  plot(NA, xlim = range(d$DAS), ylim = yl, xlab = "Days after sowing", ylab = "",
       main = if (season == S1) idx else "")
  grid0()
  for (k in 1:2) {
    col <- c(COL_D, COL_I)[k]; x <- ds[[k]]
    arrows(x$DAS, x$m - 1.96*x$se, x$DAS, x$m + 1.96*x$se, angle = 90, code = 3,
           length = 0.02, col = col, lwd = 0.8)
    lines(x$DAS, x$m, col = col, lwd = 2); points(x$DAS, x$m, pch = 19, cex = 0.6, col = col)
  }
  if (idx == INDICES[1]) mtext(paste(season, "\nobserved index"), side = 2, line = 2.6,
                               cex = 0.7, font = 2)
}
mtext("Observed vegetation index trajectories (mean +/- 95% CI, not fitted)",
      outer = TRUE, line = 0.8, font = 2, cex = 1.1)
dev.off()

# --- P8: ground time series ------------------------------------------------
gvars <- c("Chlorophyll", "Plant_height_cm", "Soil_Temp_C", "Soil_Moisture_pct")
open_png("R_P8_ground_timeseries", 2100, 1100, 150)
par(mfrow = c(2, 4), oma = c(0, 2.4, 3, 0)); PAR()
for (season in c(S1, S2)) for (v in gvars) {
  d <- ground[ground$Season == season, ]
  if (all(is.na(d[[v]]))) {
    plot.new(); title(main = if (season == S1) v else "")
    text(0.5, 0.5, paste("not recorded\nin", season), col = "grey50", cex = 0.9)
  } else {
    ds <- lapply(c("D", "I"), function(wr) {
      s <- d[d$Water_Regime == wr, ]
      k <- !is.na(s[[v]])
      data.frame(DAS = sort(unique(s$DAS[k])),
                 m = tapply(s[[v]][k], s$DAS[k], mean),
                 se = tapply(s[[v]][k], s$DAS[k], se))
    })
    plot(NA, xlim = range(d$DAS), ylim = range(unlist(lapply(ds, `[[`, "m")), na.rm = TRUE),
         xlab = "Days after sowing", ylab = "", main = if (season == S1) v else "")
    grid0()
    for (k in 1:2) { x <- ds[[k]]; col <- c(COL_D, COL_I)[k]
      lines(x$DAS, x$m, col = col, lwd = 2); points(x$DAS, x$m, pch = 19, cex = 0.6, col = col) }
  }
  if (v == gvars[1]) mtext(season, side = 2, line = 2.8, cex = 0.8, font = 2)
}
mtext("Ground measurements over time by water regime", outer = TRUE, line = 0.8,
      font = 2, cex = 1.1)
dev.off()

# --- D1: treatment response ------------------------------------------------
traits <- c("Grain_Yield_t_ha","Biological_Yield_t_ha","Harvest_Index",
            "Total_Grain_per_Plant","TSW_g","Spike_Length_cm")
open_png("R_D1_treatment_response", 2600, 1150, 150)
par(mfrow = c(2, length(traits)), oma = c(0, 2.4, 3, 0)); PAR()
for (season in c(S1, S2)) for (tr in traits) {
  d <- agroX[agroX$Season == season, ]
  ds <- lapply(c("D", "I"), function(wr) {
    s <- d[d$Water_Regime == wr, ]
    data.frame(N = sort(unique(s$N_kg_ha)),
               m = tapply(s[[tr]], s$N_kg_ha, mean, na.rm = TRUE),
               se = tapply(s[[tr]], s$N_kg_ha, se))
  })
  yl <- range(unlist(lapply(ds, function(x) c(x$m - 2*x$se, x$m + 2*x$se))), na.rm = TRUE)
  plot(NA, xlim = c(-20, 410), ylim = yl, xlab = "N (kg/ha)", ylab = "",
       main = if (season == S1) gsub("_", " ", tr) else "", xaxt = "n", cex.main = 0.95)
  axis(1, at = c(0, 210, 300, 390), cex.axis = 0.75); grid0()
  for (k in 1:2) { x <- ds[[k]]; col <- c(COL_D, COL_I)[k]
    arrows(x$N, x$m - 1.96*x$se, x$N, x$m + 1.96*x$se, angle = 90, code = 3,
           length = 0.02, col = col, lwd = 0.8)
    lines(x$N, x$m, col = col, lwd = 2); points(x$N, x$m, pch = 19, cex = 0.7, col = col) }
  if (tr == traits[1]) {
    mtext(season, side = 2, line = 2.8, cex = 0.8, font = 2)
    if (season == S1) legend("topleft", c("Drought","Irrigated"), col = c(COL_D, COL_I),
                             lwd = 2, bty = "n", cex = 0.8)
  }
}
mtext("Yield and components by nitrogen rate and water regime (mean +/- 95% CI)",
      outer = TRUE, line = 0.8, font = 2, cex = 1.1)
dev.off()

# --- D2: when to fly -------------------------------------------------------
open_png("R_D2_when_to_fly", 1900, 800, 150)
par(mfrow = c(1, 2), oma = c(0, 0, 3, 0)); PAR()
cols6 <- hcl.colors(length(INDICES), "viridis")
for (season in c(S1, S2)) {
  d <- merge(specint[specint$Season == season, ],
             harvest[, c("Plot_UID", "Grain_Yield_t_ha")], by = "Plot_UID")
  plot(NA, xlim = range(COMMON_DAS), ylim = c(-0.15, 1),
       xlab = "Days after sowing of the flight", ylab = "Pearson r with grain yield",
       main = season); grid0(); abline(h = 0, col = "black", lwd = 0.8)
  for (k in seq_along(INDICES)) {
    rs <- vapply(COMMON_DAS, function(das) {
      cc <- sprintf("%s_DAS%d", INDICES[k], das)
      if (!cc %in% names(d)) return(NA_real_)
      ok <- is.finite(d[[cc]]) & is.finite(d$Grain_Yield_t_ha)
      if (sum(ok) < 10) NA_real_ else cor(d[[cc]][ok], d$Grain_Yield_t_ha[ok])
    }, numeric(1))
    lines(COMMON_DAS, rs, col = cols6[k], lwd = 2.2)
    points(COMMON_DAS, rs, pch = 19, cex = 0.7, col = cols6[k])
  }
  if (season == S1) legend("topleft", INDICES, col = cols6, lwd = 2.2, bty = "n",
                           cex = 0.8, ncol = 2)
}
mtext("When to fly: index-to-yield correlation against flight date", outer = TRUE,
      line = 0.8, font = 2, cex = 1.1)
dev.off()

# --- D3 / D5 / D6 need the results from 02 ---------------------------------
cons <- rdr("univariate_consistency")
if (!is.null(cons)) {
  d <- cons[cons$Target == "Grain_Yield_t_ha" & cons$consistent, ]
  d <- head(d[order(-d$mean_abs_r), ], 25); d <- d[nrow(d):1, ]
  open_png("R_D3_top_univariate", 1500, 1500, 150)
  PAR(mar = c(4, 14, 3.4, 1))
  bp <- barplot(rbind(d$r_2024_25, d$r_2025_26), beside = TRUE, horiz = TRUE,
                col = c(COL_S1, COL_S2), border = NA, xlim = c(-1, 1),
                names.arg = rep("", nrow(d)), xlab = "Pearson r with grain yield")
  axis(2, at = colMeans(bp), labels = paste0(d$Parameter, "  [", d$Block, "]"),
       las = 1, cex.axis = 0.6, tick = FALSE)
  abline(v = 0, lwd = 0.8); grid0()
  legend("bottomright", c(S1, S2), fill = c(COL_S1, COL_S2), border = NA, bty = "n")
  title("Strongest associations present in BOTH seasons\n(BH-significant in each)",
        font.main = 2, cex.main = 1.0, adj = 0)
  dev.off()
}

sc <- rdr("season_comparison")
if (!is.null(sc)) {
  keep <- c("Grain_Yield_t_ha","Biological_Yield_t_ha","Harvest_Index","Fresh_Biomass_t_ha",
            "Total_Grain_per_Plant","TSW_g","Tiller_No","Spike_Length_cm","Plant_Length_cm",
            "Root_Length_cm","H2O2_umol_per_gFW","gsw","PhiPS2","ETR","Tleaf",
            "Chlorophyll_mean","Plant_height_max_cm","Soil_Temp_mean_C",
            "NDVI_AUC_common_19_83","NDVI_Peak_value","NDVI_SOS","NDVI_EOS",
            "NDVI_Green_Duration","NDVI_Amplitude")
  d <- sc[sc$Parameter %in% keep, ]; d <- d[order(d$hedges_g), ]
  open_png("R_D5_season_shift", 1400, 1400, 150)
  PAR(mar = c(4.5, 12, 3.4, 1))
  barplot(d$hedges_g, horiz = TRUE, names.arg = gsub("_", " ", d$Parameter), las = 1,
          col = ifelse(d$hedges_g < 0, COL_S2, COL_S1), border = NA, cex.names = 0.62,
          xlab = "Hedges' g  (+ = higher in 2024-25, - = higher in 2025-26)")
  abline(v = c(-0.8, -0.5, 0.5, 0.8), col = "grey70", lty = 3); abline(v = 0, lwd = 0.8)
  title("How much each parameter shifted between seasons", font.main = 2, adj = 0)
  dev.off()
}

av <- rdr("treatment_effects")
if (!is.null(av)) {
  terms <- c("Water","N","Seaweed","WxN","WxS","NxS")
  open_png("R_D6_treatment_effects", 1700, 750, 150)
  par(mfrow = c(1, 2), oma = c(0, 0, 3, 0)); PAR()
  for (season in c(S1, S2)) {
    g <- av[av$Season == season, ]
    cnt <- vapply(terms, function(t) sum(g[[paste0("q_", t)]] < 0.05, na.rm = TRUE), numeric(1))
    pct <- 100 * cnt / nrow(g)
    bp <- barplot(pct, names.arg = terms, col = c("#2B6CB0","#38A169","#D69E2E",
                  "#A0AEC0","#A0AEC0","#A0AEC0"), border = NA, las = 2,
                  ylab = "% of parameters with q < 0.05", ylim = c(0, max(pct) * 1.18),
                  main = sprintf("%s  (n = %d parameters)", season, nrow(g)))
    text(bp, pct + max(pct) * 0.04, cnt, cex = 0.8, font = 2); grid0()
  }
  mtext("Which experimental factors actually moved the measured parameters?",
        outer = TRUE, line = 0.8, font = 2, cex = 1.1)
  dev.off()
}

# --- D4: correlation matrix ------------------------------------------------
keys <- c("Grain_Yield_t_ha","Biological_Yield_t_ha","Harvest_Index","Fresh_Biomass_t_ha",
          "Total_Grain_per_Plant","TSW_g","Tiller_No","Spike_Length_cm","Plant_Length_cm",
          "Root_Length_cm","Chlorophyll_mean","Plant_height_max_cm","Soil_Temp_mean_C",
          "H2O2_umol_per_gFW","gsw","PhiPS2","ETR","Tleaf")
phw <- rdc("Phenology_Wide")
mat0 <- merge(agroX, phw[, c("Plot_UID", grep("^NDVI_", names(phw), value = TRUE))],
              by = "Plot_UID")
mat0 <- merge(mat0, specint[, c("Plot_UID", "NDVI_DAS70", "NDVI_DAS80")], by = "Plot_UID")
keys <- c(keys, "NDVI_DAS70", "NDVI_DAS80", "NDVI_AUC_common_19_83", "NDVI_Peak_value",
          "NDVI_SOS", "NDVI_EOS", "NDVI_Green_Duration")
keys <- intersect(keys, names(mat0))
open_png("R_D4_correlation_matrix", 2400, 1250, 150)
par(mfrow = c(1, 2), oma = c(0, 0, 3, 0)); PAR(mar = c(9, 9, 2.6, 2))
for (season in c(S1, S2)) {
  M <- cor(mat0[mat0$Season == season, keys], use = "pairwise.complete.obs")
  image(seq_along(keys), seq_along(keys), t(M[nrow(M):1, ]), col = hcl.colors(41, "RdBu", rev = TRUE),
        zlim = c(-1, 1), xlab = "", ylab = "", axes = FALSE, main = season)
  axis(1, seq_along(keys), gsub("_", " ", keys), las = 2, cex.axis = 0.55, tick = FALSE)
  axis(2, seq_along(keys), rev(gsub("_", " ", keys)), las = 1, cex.axis = 0.55, tick = FALSE)
}
mtext("Correlation structure among headline parameters", outer = TRUE, line = 0.8,
      font = 2, cex = 1.1)
dev.off()

cat("figures ->", FIGD, "\n")
cat(paste(" ", sort(list.files(FIGD, pattern = "^R_.*png$")), collapse = "\n"), "\n")
