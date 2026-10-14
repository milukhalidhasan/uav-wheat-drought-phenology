# ---------------------------------------------------------------------------
# Shared configuration: paths, study constants, the double-logistic model.
# Every other script starts with  source("00_config.R")
#
# Dependencies: BASE R ONLY.
#   readxl is needed only by 01_build_combined_dataset.R
#   the ML packages are needed only by 03_ml_analysis.R
#   statistics (02) and figures (04) run on a bare R installation.
#
# Project root is inferred from this file's location:
#
#   <PROJECT_ROOT>/
#       Wheat_Analysis_2024/05_Dataset/Wheat_2024-25_Structured_Dataset.xlsx
#       Wheat_Analysis_2025/MegaData2025_Cleaned.xlsx
#       Combined_TwoSeason/06_R_Code/00_config.R   <- this file
#
# Override with:  Sys.setenv(WHEAT_ROOT = "D:/my/project")
# ---------------------------------------------------------------------------

this_file <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", a, value = TRUE)
  if (length(m)) return(normalizePath(sub("^--file=", "", m[1]), mustWork = FALSE))
  fr <- sys.frames()
  for (i in rev(seq_along(fr)))
    if (!is.null(fr[[i]]$ofile)) return(normalizePath(fr[[i]]$ofile, mustWork = FALSE))
  file.path(getwd(), "00_config.R")
}

R_DIR    <- dirname(this_file())
COMBINED <- dirname(R_DIR)
ROOT     <- Sys.getenv("WHEAT_ROOT", unset = dirname(COMBINED))

F24 <- Sys.getenv("WHEAT_S1_XLSX", unset = file.path(
  ROOT, "Wheat_Analysis_2024", "05_Dataset", "Wheat_2024-25_Structured_Dataset.xlsx"))
F25 <- Sys.getenv("WHEAT_S2_XLSX", unset = file.path(
  ROOT, "Wheat_Analysis_2025", "MegaData2025_Cleaned.xlsx"))

DATASET_DIR <- file.path(COMBINED, "02_Dataset")
CSVD        <- file.path(COMBINED, "03_CSV")
ML_DIR      <- file.path(COMBINED, "04_ML")
RESD        <- file.path(ML_DIR, "results_R")   # kept separate from the Python results
FIGD        <- file.path(ML_DIR, "figures_R")   # kept separate from the Python figures

for (d in c(DATASET_DIR, CSVD, ML_DIR, RESD, FIGD))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)

check_inputs <- function() {
  miss <- c(F24, F25)[!file.exists(c(F24, F25))]
  if (length(miss))
    stop("Source workbook(s) not found:\n  ", paste(miss, collapse = "\n  "),
         "\n\nSet WHEAT_ROOT (or WHEAT_S1_XLSX / WHEAT_S2_XLSX).", call. = FALSE)
  invisible(TRUE)
}

# ---------------------------------------------------------------------------
# study constants (identical to the Python pipeline)
# ---------------------------------------------------------------------------
S1 <- "2024-25"; S2 <- "2025-26"
FLIGHT_WINDOW <- list(`2024-25` = c(10, 108), `2025-26` = c(19, 83))
COMMON_WINDOW <- c(19, 83)
COMMON_DAS    <- c(20, 30, 40, 50, 60, 70, 80)
LN9 <- log(9); SEED <- 20260919

INDICES <- c("NDVI", "NDRE", "CI(rededge)", "GCI", "gNDVI", "RVI")
TARGETS <- c("Grain_Yield_t_ha", "Biological_Yield_t_ha", "Harvest_Index")

# T0-T11 = 4 nitrogen levels x 3 seaweed levels, nitrogen varying slowest
N_DOSE <- c("0" = 0, "1" = 210, "2" = 300, "3" = 390)
N_DESC <- c("0" = "Control", "1" = "Low", "2" = "Medium", "3" = "High")
S_DOSE <- c("0" = 0, "1" = 4, "2" = 10)
S_DESC <- c("0" = "No application", "1" = "Medium", "2" = "High")

COMPONENTS <- c("Tiller_No", "Spike_Length_cm", "Total_Grain_per_Plant", "TSW_g",
                "Fresh_Biomass_t_ha")
FORBIDDEN <- list(
  Grain_Yield_t_ha      = c("Biological_Yield_t_ha", "Harvest_Index"),
  Biological_Yield_t_ha = c("Grain_Yield_t_ha", "Harvest_Index"),
  Harvest_Index         = c("Grain_Yield_t_ha", "Biological_Yield_t_ha"))

DESIGN_FEATURES <- c("N_Level", "N_kg_ha", "Seaweed_Level", "Seaweed_pct")
ML_KEYS <- c("Season", "Season_ID", "Plot_UID", "Plot_ID", "Water_Regime", "Treatment",
             DESIGN_FEATURES, "Replication")
DESIGN_CAT <- c("Water_Regime", "N_Level", "Seaweed_Level")

# plotting colours
COL_D  <- "#C05621"; COL_I  <- "#2B6CB0"    # drought / irrigated
COL_S1 <- "#2B6CB0"; COL_S2 <- "#C05621"    # season 1 / season 2

# ---------------------------------------------------------------------------
# the double-logistic model in the harmonised parameterisation
#
#   y(t) = Baseline + Amplitude * [ sigma(k1 (t - t1)) - sigma(k2 (t - t2)) ]
#
#   2024-25 source names : Baseline = a,  Amplitude = b (= e), t1 = d, t2 = g,
#                          k1 = c, k2 = f
#   2025-26 source names : Baseline = y0, Amplitude = y1 - y0, t1 = t1, t2 = t2,
#                          k1 = k1, k2 = k2   <- y1 is the PEAK, not the amplitude
# ---------------------------------------------------------------------------
sigmoid <- function(x) 1 / (1 + exp(-pmax(pmin(x, 500), -500)))

dlogistic <- function(t, base, amp, t1, t2, k1, k2)
  base + amp * (sigmoid(k1 * (t - t1)) - sigmoid(k2 * (t - t2)))

CURVE_STAT_NAMES <- c("SOS","EOG","POS","SOD","EOS","Peak_value","Greenup_Duration",
                      "Plateau_Duration","Senescence_Duration","Green_Duration","LOS",
                      "Rate_greenup","Rate_senescence","AUC_native","AUC_common_19_83",
                      "AUC_above_base_common","AUC_pheno_SOS_EOS","AUC_common_per_day",
                      "n_stages_outside_common","n_stages_outside_native")

curve_stats <- function(base, amp, t1, t2, k1, k2, season) {
  f <- function(t) dlogistic(t, base, amp, t1, t2, k1, k2)
  SOS <- t1 - LN9 / k1; EOG <- t1 + LN9 / k1
  SOD <- t2 - LN9 / k2; EOS <- t2 + LN9 / k2
  op  <- optimize(f, c(min(SOS, t1) - 5, max(EOS, t2) + 5), maximum = TRUE, tol = 1e-8)
  POS <- op$maximum; peak <- op$objective
  wn <- FLIGHT_WINDOW[[season]]; wc <- COMMON_WINDOW
  ii <- function(lo, hi) tryCatch(
    integrate(f, lo, hi, subdivisions = 500, rel.tol = 1e-9)$value,
    error = function(e) NA_real_)
  auc_n <- ii(wn[1], wn[2]); auc_c <- ii(wc[1], wc[2]); auc_p <- ii(SOS, EOS)
  st <- c(SOS, EOG, POS, SOD, EOS)
  out <- c(SOS, EOG, POS, SOD, EOS, peak,
           EOG - SOS, SOD - EOG, EOS - SOD, t2 - t1, EOS - SOS,
           amp * k1 / 4, amp * k2 / 4,
           auc_n, auc_c, auc_c - base * diff(wc), auc_p, auc_c / diff(wc),
           sum(st < wc[1] | st > wc[2]), sum(st < wn[1] | st > wn[2]))
  names(out) <- CURVE_STAT_NAMES
  out
}

# ---------------------------------------------------------------------------
# small helpers (base R)
# ---------------------------------------------------------------------------
has_pkg <- function(p) requireNamespace(p, quietly = TRUE)

need <- function(pkgs) {
  miss <- pkgs[!vapply(pkgs, has_pkg, logical(1))]
  if (length(miss))
    stop("Missing R packages: ", paste(miss, collapse = ", "),
         "\nRun:  source('install_packages.R')", call. = FALSE)
  invisible(TRUE)
}

r2_score <- function(y, yhat) {
  ss <- sum((y - mean(y))^2)
  if (ss <= 0) return(NA_real_)
  1 - sum((y - yhat)^2) / ss
}
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
mae  <- function(y, yhat) mean(abs(y - yhat))

read_tbl  <- function(n) read.csv(file.path(CSVD, paste0(n, ".csv")),
                                  check.names = FALSE, stringsAsFactors = FALSE)
write_tbl <- function(d, n) write.csv(d, file.path(RESD, paste0(n, ".csv")),
                                      row.names = FALSE, na = "")

open_png <- function(name, w = 1800, h = 1100, res = 150) {
  png(file.path(FIGD, paste0(name, ".png")), width = w, height = h, res = res,
      type = "cairo-png", bg = "white")
}

cat("config loaded | ROOT =", ROOT, "\n")
