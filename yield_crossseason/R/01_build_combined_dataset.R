# ---------------------------------------------------------------------------
# Build the harmonised two-season dataset from the two source workbooks.
# R port of 01_Code/build_combined_dataset.py — same tables, same values.
#
# Needs: readxl  (writexl optional, for the .xlsx output)
# Run:   Rscript 01_build_combined_dataset.R
# ---------------------------------------------------------------------------
source("00_config.R")
need("readxl")
check_inputs()

rd <- function(f, s) as.data.frame(readxl::read_excel(f, sheet = s), check.names = FALSE)

ren <- function(df, from, to) { names(df)[names(df) == from] <- to; df }

season_cols <- function(df, season) {
  sid <- if (season == S1) 1L else 2L
  out <- data.frame(Season = season, Season_ID = sid, df,
                    check.names = FALSE, stringsAsFactors = FALSE)
  if ("Plot_ID" %in% names(df)) {
    out <- data.frame(out[, 1:2, drop = FALSE],
                      Plot_UID = paste0("S", sid, "_", df$Plot_ID),
                      out[, -(1:2), drop = FALSE],
                      check.names = FALSE, stringsAsFactors = FALSE)
  }
  out
}

# bind two frames, filling missing columns with NA, preserving column order
rbind_fill <- function(a, b, cols) {
  for (cc in cols) {
    if (!cc %in% names(a)) a[[cc]] <- NA
    if (!cc %in% names(b)) b[[cc]] <- NA
  }
  rbind(a[, cols, drop = FALSE], b[, cols, drop = FALSE])
}

# ---------------------------------------------------------------------------
# 1. plot design, with the treatment codes decoded
# ---------------------------------------------------------------------------
build_design <- function() {
  d24 <- rd(F24, "Plot_Design")
  d24 <- ren(ren(d24, "Plot", "Field_Plot_No"), "dCHM_relative", "dCHM_relative_plot")
  d24 <- season_cols(d24, S1)

  h25 <- rd(F25, "Harvest_PlotLevel")
  d25 <- h25[, c("Plot_ID", "Water_Regime", "Treatment", "Replication")]
  d25$Field_Plot_No <- NA_real_; d25$dCHM_relative_plot <- NA_real_
  d25 <- season_cols(d25, S2)

  cols <- c("Season","Season_ID","Plot_UID","Plot_ID","Field_Plot_No","Water_Regime",
            "Treatment","Replication","dCHM_relative_plot")
  out <- rbind_fill(d24, d25, cols)
  out$Treatment_Num      <- as.integer(sub("T", "", out$Treatment))
  out$Water_Regime_Label <- ifelse(out$Water_Regime == "D", "Managed drought", "Irrigated")
  ni <- out$Treatment_Num %/% 3; si <- out$Treatment_Num %% 3
  out$N_Level       <- paste0("N", ni)
  out$N_kg_ha       <- unname(N_DOSE[as.character(ni)])
  out$N_Desc        <- unname(N_DESC[as.character(ni)])
  out$Seaweed_Level <- paste0("S", si)
  out$Seaweed_pct   <- unname(S_DOSE[as.character(si)])
  out$Seaweed_Desc  <- unname(S_DESC[as.character(si)])
  out <- out[order(out$Season_ID, out$Plot_ID), ]
  out[, c("Season","Season_ID","Plot_UID","Plot_ID","Field_Plot_No","Water_Regime",
          "Water_Regime_Label","Treatment","Treatment_Num","N_Level","N_kg_ha","N_Desc",
          "Seaweed_Level","Seaweed_pct","Seaweed_Desc","Replication","dCHM_relative_plot")]
}

# ---------------------------------------------------------------------------
# 2. harvest (2025-26 stores leaf physiology in this sheet; split it out)
# ---------------------------------------------------------------------------
HARVEST_ORDER <- c("Season","Season_ID","Plot_UID","Plot_ID","Water_Regime","Treatment",
                   "Replication","Plant_Length_cm","Root_Length_cm","Tiller_No",
                   "Spike_Length_cm","Total_Grain_per_Plant","TSW_g","Fresh_Biomass_t_ha",
                   "Biological_Yield_t_ha","Grain_Yield_t_ha","Harvest_Index",
                   "H2O2_umol_per_gFW","MDA_nmol_per_gFW")

build_harvest <- function() {
  h24 <- rd(F24, "Harvest_PlotLevel"); h24$Plot <- NULL
  h24 <- season_cols(h24, S1)

  h25 <- rd(F25, "Harvest_PlotLevel")
  map <- c("1000_Seed_Weight_(g)" = "TSW_g",
           "Fresh_Biomass(t/ha)"  = "Fresh_Biomass_t_ha",
           "Biological_Yield(t/ha)" = "Biological_Yield_t_ha",
           "Grain_Yield(t/ha)"    = "Grain_Yield_t_ha",
           "Harvest_Index(HI)"    = "Harvest_Index",
           "H2O2 (umol/g FW)"     = "H2O2_umol_per_gFW")
  for (a in names(map)) h25 <- ren(h25, a, map[[a]])
  phys25 <- h25[, c("Plot_ID", "gsw", "PhiPS2", "ETR", "Tleaf")]
  h25 <- h25[, setdiff(names(h25), c("gsw", "PhiPS2", "ETR", "Tleaf"))]
  h25 <- season_cols(h25, S2)

  out <- rbind_fill(h24, h25, HARVEST_ORDER)
  out <- out[order(out$Season_ID, out$Plot_ID), ]
  list(harvest = out, phys25 = phys25)
}

# ---------------------------------------------------------------------------
# 3. leaf physiology: 2024-25 measured at 80 and 88 DAS, averaged here
# ---------------------------------------------------------------------------
build_physiology <- function(phys25) {
  lp <- rd(F24, "Leaf_Physiology")
  idc  <- c("Plot_ID", "Water_Regime", "Replication", "Treatment", "DAS")
  vars <- setdiff(names(lp), idc)
  m <- aggregate(lp[, vars], by = list(Plot_ID = lp$Plot_ID), FUN = mean, na.rm = TRUE)

  for (v in c("gsw", "PhiPS2", "ETR", "Tleaf")) {
    for (d in sort(unique(lp$DAS))) {
      sub <- lp[lp$DAS == d, c("Plot_ID", v)]
      names(sub)[2] <- sprintf("%s_DAS%d", v, d)
      m <- merge(m, sub, by = "Plot_ID", all.x = TRUE)
    }
  }
  p24 <- season_cols(m, S1); p24$Physiology_Source <- "mean of 80 and 88 DAS"
  p25 <- season_cols(as.data.frame(phys25), S2)
  p25$Physiology_Source <- "single measurement (harvest sheet)"

  cols <- union(names(p24), names(p25))
  out  <- rbind_fill(p24, p25, cols)
  lead <- c("Season","Season_ID","Plot_UID","Plot_ID","Physiology_Source",
            "gsw","PhiPS2","ETR","Tleaf")
  out <- out[order(out$Season_ID, out$Plot_ID), c(lead, setdiff(names(out), lead))]
  out
}

# ---------------------------------------------------------------------------
# 4. ground time series and its plot-level summary
# ---------------------------------------------------------------------------
GROUND_COLS <- c("Season","Season_ID","Plot_UID","Plot_ID","Water_Regime","Replication",
                 "Treatment","DAS","Growth_Stage","Chlorophyll","Plant_height_cm",
                 "Soil_Temp_C","Soil_Moisture_pct")

build_ground <- function() {
  g24 <- rd(F24, "Ground_TimeSeries")
  g24 <- ren(ren(ren(g24, "Plant_height", "Plant_height_cm"),
                 "Soil_Temp", "Soil_Temp_C"), "Soil_Moisture", "Soil_Moisture_pct")
  g24$Growth_Stage <- NA_character_
  g24 <- season_cols(g24, S1)

  g25 <- rd(F25, "Ground_TimeSeries")
  g25$Soil_Moisture_pct <- NA_real_
  g25 <- season_cols(g25, S2)

  out <- rbind_fill(g24, g25, GROUND_COLS)
  out[order(out$Season_ID, out$Plot_ID, out$DAS), ]
}

last_valid <- function(v) { v <- v[!is.na(v)]; if (length(v)) v[length(v)] else NA_real_ }
safe <- function(f) function(v) { v <- v[!is.na(v)]; if (length(v)) f(v) else NA_real_ }

summarise_ground <- function(g) {
  g   <- g[order(g$DAS), ]
  key <- unique(g[, c("Season","Season_ID","Plot_UID","Plot_ID")])
  agg <- function(col, f) vapply(key$Plot_UID,
                                 function(u) f(g[[col]][g$Plot_UID == u]), numeric(1))
  data.frame(key,
    Chlorophyll_mean       = agg("Chlorophyll",      safe(mean)),
    Chlorophyll_max        = agg("Chlorophyll",      safe(max)),
    Chlorophyll_last       = agg("Chlorophyll",      last_valid),
    Plant_height_max_cm    = agg("Plant_height_cm",  safe(max)),
    Plant_height_mean_cm   = agg("Plant_height_cm",  safe(mean)),
    Soil_Temp_mean_C       = agg("Soil_Temp_C",      safe(mean)),
    Soil_Moisture_mean_pct = agg("Soil_Moisture_pct", safe(mean)),
    n_ground_dates         = agg("DAS", function(v) length(unique(v))),
    row.names = NULL, check.names = FALSE)
}

# ---------------------------------------------------------------------------
# 5. spectral series and interpolation onto the common DAS grid
# ---------------------------------------------------------------------------
SPEC_LEAD <- c("Season","Season_ID","Plot_UID","Plot_ID","Water_Regime","Replication",
               "Treatment","Flight_Date","DAS")
BANDS <- c("blue","green","red","rededge","nir","Thermal","dchm")
SEASON_ONLY_SPECTRAL <- c("Thermal", "dchm", "SR")

build_spectral <- function() {
  s24 <- ren(rd(F24, "Spectral_TimeSeries"), "DOY", "Flight_Date")
  s24$Growth_Stage <- NA_character_; s24$Thermal <- NA_real_; s24$dchm <- NA_real_
  s24 <- season_cols(s24, S1)

  s25 <- ren(rd(F25, "Spectral_TimeSeries"), "DOY", "Flight_Date")
  s25 <- s25[, setdiff(names(s25), c("L_coef", "C1_coef", "C2_coef"))]  # constants
  s25$SR <- NA_real_
  s25 <- season_cols(s25, S2)

  meta <- c(SPEC_LEAD, BANDS, "Growth_Stage")
  vis  <- setdiff(names(s25), c(meta, "SR"))
  vis  <- c(intersect(vis, names(s24)), "SR")
  cols <- c(SPEC_LEAD, "Growth_Stage", BANDS, vis)
  out  <- rbind_fill(s24, s25, cols)
  out$Flight_Date  <- as.Date(out$Flight_Date)
  out$DOY_calendar <- as.integer(format(out$Flight_Date, "%j"))
  out <- out[order(out$Season_ID, out$Plot_ID, out$DAS), ]
  list(spectral = out, vis = vis)
}

interp_common <- function(spec, vis) {
  value_cols <- setdiff(c(BANDS, vis), SEASON_ONLY_SPECTRAL)
  keys <- c("Season","Season_ID","Plot_UID","Plot_ID","Water_Regime","Replication","Treatment")
  uids <- unique(spec$Plot_UID)
  out  <- spec[match(uids, spec$Plot_UID), keys]
  M <- matrix(NA_real_, nrow = length(uids),
              ncol = length(value_cols) * length(COMMON_DAS))
  nm <- character(ncol(M)); j <- 0L
  for (v in value_cols) for (d in COMMON_DAS) { j <- j + 1L; nm[j] <- sprintf("%s_DAS%d", v, d) }
  colnames(M) <- nm
  for (i in seq_along(uids)) {
    g <- spec[spec$Plot_UID == uids[i], ]; g <- g[order(g$DAS), ]
    j <- 0L
    for (v in value_cols) {
      y <- suppressWarnings(as.numeric(g[[v]])); x <- g$DAS
      ok <- is.finite(y) & is.finite(x)
      vals <- if (sum(ok) >= 2) approx(x[ok], y[ok], xout = COMMON_DAS, rule = 2)$y
              else rep(NA_real_, length(COMMON_DAS))
      M[i, (j + 1L):(j + length(COMMON_DAS))] <- vals
      j <- j + length(COMMON_DAS)
    }
  }
  out <- data.frame(out, M, check.names = FALSE, row.names = NULL)
  out[order(out$Season_ID, out$Plot_ID), ]
}

# ---------------------------------------------------------------------------
# 6. phenology: map both parameterisations onto one, then recompute the stages
# ---------------------------------------------------------------------------
PHENO_KEEP <- c("Season","Season_ID","Plot_UID","Plot_ID","Water_Regime","Replication",
                "Treatment","Index","Param_Notation_Source","Baseline","Amplitude",
                "Peak_asymptote","t1_mid_greenup","t2_mid_senescence","k1_rate_greenup",
                "k2_rate_senescence","R2","RMSE","Params_At_Bound","AUC_source")

build_phenology <- function() {
  p24 <- rd(F24, "Phenology_Parameters")
  p24$Baseline <- p24$a; p24$Amplitude <- p24$b; p24$Peak_asymptote <- p24$a + p24$b
  p24$t1_mid_greenup <- p24$d; p24$t2_mid_senescence <- p24$g
  p24$k1_rate_greenup <- p24$c; p24$k2_rate_senescence <- p24$f
  p24$AUC_source <- p24$AUC; p24$Params_At_Bound <- p24$n_params_at_bound
  p24 <- season_cols(p24, S1); p24$Param_Notation_Source <- "a,b,c,d,e,f,g (b = e)"

  p25 <- rd(F25, "Phenology_Parameters")
  p25$Baseline <- p25$y0; p25$Amplitude <- p25$y1 - p25$y0; p25$Peak_asymptote <- p25$y1
  p25$t1_mid_greenup <- p25$t1; p25$t2_mid_senescence <- p25$t2
  p25$k1_rate_greenup <- p25$k1; p25$k2_rate_senescence <- p25$k2
  p25$AUC_source <- p25$AUC
  p25 <- season_cols(p25, S2); p25$Param_Notation_Source <- "y0,y1,t1,t2,k1,k2 (y1 = peak)"

  out <- rbind_fill(p24, p25, PHENO_KEEP)
  der <- t(vapply(seq_len(nrow(out)), function(i)
    curve_stats(out$Baseline[i], out$Amplitude[i], out$t1_mid_greenup[i],
                out$t2_mid_senescence[i], out$k1_rate_greenup[i],
                out$k2_rate_senescence[i], out$Season[i]),
    numeric(length(CURVE_STAT_NAMES))))
  out <- data.frame(out, as.data.frame(der), check.names = FALSE, row.names = NULL)
  out$Fit_Quality_Flag <- ifelse(out$R2 < 0.80, "poor",
                          ifelse(out$R2 < 0.90 | out$Params_At_Bound > 0, "check", "ok"))
  out[order(out$Season_ID, out$Index, out$Plot_ID), ]
}

PHENO_ML_PARAMS <- c("Baseline","Amplitude","Peak_asymptote","Peak_value",
                     "t1_mid_greenup","t2_mid_senescence","k1_rate_greenup",
                     "k2_rate_senescence","SOS","EOG","POS","SOD","EOS",
                     "Greenup_Duration","Plateau_Duration","Senescence_Duration",
                     "Green_Duration","LOS","Rate_greenup","Rate_senescence",
                     "AUC_common_19_83","AUC_above_base_common","AUC_pheno_SOS_EOS","R2")

widen_phenology <- function(ph) {
  keys <- c("Season","Season_ID","Plot_UID","Plot_ID","Water_Regime","Replication","Treatment")
  uids <- unique(ph$Plot_UID)
  out  <- ph[match(uids, ph$Plot_UID), keys]
  idx  <- sort(unique(ph$Index))
  nm <- unlist(lapply(idx, function(i)
    paste0(gsub("[()]", "", i), "_", PHENO_ML_PARAMS)))
  M <- matrix(NA_real_, nrow = length(uids), ncol = length(nm), dimnames = list(NULL, nm))
  for (i in idx) {
    sub <- ph[ph$Index == i, ]
    pos <- match(uids, sub$Plot_UID)
    for (p in PHENO_ML_PARAMS)
      M[, paste0(gsub("[()]", "", i), "_", p)] <- sub[[p]][pos]
  }
  M <- M[, sort(colnames(M)), drop = FALSE]
  out <- data.frame(out, M, check.names = FALSE, row.names = NULL)
  out[order(out$Season_ID, out$Plot_ID), ]
}

# ---------------------------------------------------------------------------
# 7. ML-ready tables
# ---------------------------------------------------------------------------
AGRONOMIC        <- c("Plant_Length_cm","Root_Length_cm","Fresh_Biomass_t_ha")
YIELD_COMPONENTS <- c("Tiller_No","Spike_Length_cm","Total_Grain_per_Plant","TSW_g")
STRESS           <- c("H2O2_umol_per_gFW","MDA_nmol_per_gFW")
PHYSIOLOGY       <- c("gsw","PhiPS2","ETR","Tleaf")
GROUND_SUMMARY   <- c("Chlorophyll_mean","Chlorophyll_max","Chlorophyll_last",
                      "Plant_height_max_cm","Plant_height_mean_cm","Soil_Temp_mean_C",
                      "Soil_Moisture_mean_pct")

build_ml_agronomic <- function(design, harvest, phys, gsum) {
  d <- design[, setdiff(names(design), c("Field_Plot_No","Water_Regime_Label",
                                         "Treatment_Num","N_Desc","Seaweed_Desc"))]
  hv <- harvest[, setdiff(names(harvest), c("Season","Season_ID","Water_Regime",
                                            "Treatment","Replication","Plot_ID"))]
  d <- merge(d, hv, by = "Plot_UID", all.x = TRUE)
  d <- merge(d, phys[, c("Plot_UID", PHYSIOLOGY)], by = "Plot_UID", all.x = TRUE)
  d <- merge(d, gsum[, c("Plot_UID", GROUND_SUMMARY)], by = "Plot_UID", all.x = TRUE)
  feats <- c(AGRONOMIC, YIELD_COMPONENTS, STRESS, PHYSIOLOGY, GROUND_SUMMARY,
             "dCHM_relative_plot")
  d <- d[, c(ML_KEYS, intersect(feats, names(d)), TARGETS)]
  d[order(d$Season_ID, d$Plot_ID), ]
}

build_ml_phenology <- function(phw, harvest, design) {
  d <- merge(phw, design[, c("Plot_UID", DESIGN_FEATURES)], by = "Plot_UID", all.x = TRUE)
  d <- merge(d, harvest[, c("Plot_UID", TARGETS)], by = "Plot_UID", all.x = TRUE)
  rest <- setdiff(names(d), c(ML_KEYS, TARGETS))
  d <- d[, c(ML_KEYS, rest, TARGETS)]
  d[order(d$Season_ID, d$Plot_ID), ]
}

drop_season_specific <- function(df) {
  out <- df[, setdiff(names(df), c("MDA_nmol_per_gFW","Soil_Moisture_mean_pct",
                                   "dCHM_relative_plot"))]
  keep <- vapply(names(out), function(c0) {
    if (c0 %in% ML_KEYS) return(TRUE)
    all(tapply(out[[c0]], out$Season, function(v) any(!is.na(v))))
  }, logical(1))
  out[, keep, drop = FALSE]
}

# ---------------------------------------------------------------------------
main <- function() {
  cat("harmonising design / harvest / physiology / ground ...\n")
  design <- build_design()
  hv <- build_harvest(); harvest <- hv$harvest
  phys <- build_physiology(hv$phys25)
  ground <- build_ground(); gsum <- summarise_ground(ground)

  cat("harmonising spectral series ...\n")
  sp <- build_spectral()
  spec_interp <- interp_common(sp$spectral, sp$vis)

  cat("harmonising phenology and recomputing derived stages ...\n")
  pheno <- build_phenology(); phw <- widen_phenology(pheno)

  cat("assembling ML tables ...\n")
  ml_agro <- build_ml_agronomic(design, harvest, phys, gsum)
  ml_phen <- build_ml_phenology(phw, harvest, design)

  tables <- list(
    Plot_Design = design, Harvest_PlotLevel = harvest, Physiology_PlotLevel = phys,
    Ground_TimeSeries = ground, Ground_PlotSummary = gsum,
    Spectral_TimeSeries = sp$spectral, Spectral_Interp_CommonDAS = spec_interp,
    Phenology_Parameters = pheno, Phenology_Wide = phw,
    ML_Agronomic_vs_Yield = ml_agro, ML_Phenology_vs_Yield = ml_phen,
    ML_Agronomic_CrossSeason = drop_season_specific(ml_agro),
    ML_Phenology_CrossSeason = drop_season_specific(ml_phen))

  cat("writing CSVs ->", CSVD, "\n")
  for (n in names(tables))
    write.csv(tables[[n]], file.path(CSVD, paste0(n, "_R.csv")),
              row.names = FALSE, na = "")
  if (has_pkg("writexl")) {
    f <- file.path(DATASET_DIR, "Wheat_TwoSeason_Combined_R.xlsx")
    writexl::write_xlsx(tables, f); cat("wrote", f, "\n")
  }
  for (n in names(tables))
    cat(sprintf("  %-28s %5d rows x %4d cols\n", n, nrow(tables[[n]]), ncol(tables[[n]])))
  invisible(tables)
}

tables <- main()
