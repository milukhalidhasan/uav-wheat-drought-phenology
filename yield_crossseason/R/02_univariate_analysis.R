# ---------------------------------------------------------------------------
# Univariate screening of EVERY available parameter against every yield target,
# plus the factorial treatment analysis and the season comparison.
# R port of 01_Code/univariate_analysis.py.
#
# Dependencies: BASE R ONLY.
# Run: Rscript 02_univariate_analysis.R      (after 01_build_combined_dataset.R)
#
# Outputs (in 04_ML/results_R/):
#   univariate_correlations.csv   Pearson + Spearman per season, with BH q values
#   univariate_consistency.csv    cross-season consistency of each association
#   treatment_effects.csv         Type II 3-way ANOVA (water x N x seaweed)
#   season_comparison.csv         season means, Welch t, Hedges' g
# ---------------------------------------------------------------------------
source("00_config.R")

SUF <- if (file.exists(file.path(CSVD, "Plot_Design_R.csv"))) "_R" else ""
rdc <- function(n) read.csv(file.path(CSVD, paste0(n, SUF, ".csv")),
                            check.names = FALSE, stringsAsFactors = FALSE)

# ---------------------------------------------------------------------------
# assemble every parameter into one plot-season matrix, tagged by block
# ---------------------------------------------------------------------------
build_matrix <- function() {
  design  <- rdc("Plot_Design")
  harvest <- rdc("Harvest_PlotLevel")
  physio  <- rdc("Physiology_PlotLevel")
  gsum    <- rdc("Ground_PlotSummary")
  ground  <- rdc("Ground_TimeSeries")
  pheno   <- rdc("ML_Phenology_CrossSeason")
  spec    <- rdc("Spectral_Interp_CommonDAS")

  # ground data pivoted to one column per variable per measurement date
  gvars <- c("Chlorophyll", "Plant_height_cm", "Soil_Temp_C", "Soil_Moisture_pct")
  uids  <- unique(ground$Plot_UID)
  gts   <- data.frame(Plot_UID = uids, stringsAsFactors = FALSE)
  for (v in gvars) for (d in sort(unique(ground$DAS))) {
    sub <- ground[ground$DAS == d, c("Plot_UID", v)]
    col <- sprintf("%s_D%d", v, d)
    gts[[col]] <- sub[[v]][match(uids, sub$Plot_UID)]
  }
  gts <- gts[, c(TRUE, vapply(gts[-1], function(v) any(!is.na(v)), logical(1)))]

  drop <- function(d, cols) d[, setdiff(names(d), cols), drop = FALSE]
  df <- design[, c("Season","Season_ID","Plot_UID","Plot_ID","Water_Regime","Treatment",
                   "N_Level","N_kg_ha","Seaweed_Level","Seaweed_pct","Replication",
                   "dCHM_relative_plot")]
  for (part in list(
        drop(harvest, c("Season","Season_ID","Plot_ID","Water_Regime","Treatment","Replication")),
        drop(physio,  c("Season","Season_ID","Plot_ID","Physiology_Source")),
        drop(gsum,    c("Season","Season_ID","Plot_ID")),
        gts,
        drop(pheno,   c(setdiff(ML_KEYS, "Plot_UID"), TARGETS)),
        drop(spec,    c("Season","Season_ID","Plot_ID","Water_Regime","Replication","Treatment"))))
    df <- merge(df, part, by = "Plot_UID", all.x = TRUE)

  blocks <- list(
    Agro_predictive = c("Plant_Length_cm","Root_Length_cm","H2O2_umol_per_gFW"),
    Agro_components = COMPONENTS,
    Physio          = c("gsw","PhiPS2","ETR","Tleaf"),
    Physio_extra24  = c("gbw","gtw","E_apparent","VPleaf","VPDleaf","H2O_leaf","Fs","Fm.",
                        "rh_s","Qamb","gsw_DAS80","gsw_DAS88","PhiPS2_DAS80","PhiPS2_DAS88",
                        "ETR_DAS80","ETR_DAS88","Tleaf_DAS80","Tleaf_DAS88"),
    Ground_summary  = c("Chlorophyll_mean","Chlorophyll_max","Chlorophyll_last",
                        "Plant_height_max_cm","Plant_height_mean_cm","Soil_Temp_mean_C",
                        "Soil_Moisture_mean_pct"),
    Ground_ts       = setdiff(names(gts), "Plot_UID"),
    Stress          = "MDA_nmol_per_gFW",
    Structure24     = "dCHM_relative_plot",
    Pheno           = grep("_R2$", setdiff(names(pheno), c(ML_KEYS, TARGETS)),
                           value = TRUE, invert = TRUE),
    Spectral        = setdiff(names(spec), c("Season","Season_ID","Plot_UID","Plot_ID",
                                             "Water_Regime","Replication","Treatment")))
  blocks <- lapply(blocks, function(v) intersect(v, names(df)))
  list(df = df[order(df$Season_ID, df$Plot_ID), ], blocks = blocks)
}

block_of <- function(col, blocks) {
  for (n in names(blocks)) if (col %in% blocks[[n]]) return(n)
  "other"
}

# ---------------------------------------------------------------------------
# 1. correlations, with Benjamini-Hochberg within each season x target
# ---------------------------------------------------------------------------
correlations <- function(df, blocks, params) {
  rows <- list(); k <- 0L
  for (tgt in TARGETS) for (season in c(S1, S2)) {
    d <- df[df$Season == season, ]
    y <- suppressWarnings(as.numeric(d[[tgt]]))
    for (p in params) {
      if (p == tgt || p %in% FORBIDDEN[[tgt]]) next
      x <- suppressWarnings(as.numeric(d[[p]]))
      ok <- is.finite(x) & is.finite(y)
      if (sum(ok) < 10 || sd(x[ok]) == 0) next
      ct <- suppressWarnings(cor.test(x[ok], y[ok], method = "pearson"))
      st <- suppressWarnings(cor.test(x[ok], y[ok], method = "spearman", exact = FALSE))
      k <- k + 1L
      rows[[k]] <- data.frame(Target = tgt, Season = season, Parameter = p,
                              Block = block_of(p, blocks), n = sum(ok),
                              pearson_r = unname(ct$estimate), pearson_p = ct$p.value,
                              spearman_rho = unname(st$estimate), spearman_p = st$p.value,
                              stringsAsFactors = FALSE)
    }
  }
  out <- do.call(rbind, rows)
  out$pearson_q_BH <- NA_real_; out$spearman_q_BH <- NA_real_
  for (tgt in TARGETS) for (season in c(S1, S2)) {
    m <- out$Target == tgt & out$Season == season
    if (any(m)) {
      out$pearson_q_BH[m]  <- p.adjust(out$pearson_p[m],  "BH")
      out$spearman_q_BH[m] <- p.adjust(out$spearman_p[m], "BH")
    }
  }
  out
}

consistency <- function(corr) {
  key <- unique(corr[, c("Target", "Parameter", "Block")])
  g <- function(tgt, par, season, col) {
    v <- corr[[col]][corr$Target == tgt & corr$Parameter == par & corr$Season == season]
    if (length(v)) v[1] else NA_real_
  }
  key$r_2024_25 <- mapply(g, key$Target, key$Parameter, S1, "pearson_r")
  key$r_2025_26 <- mapply(g, key$Target, key$Parameter, S2, "pearson_r")
  key$q_2024_25 <- mapply(g, key$Target, key$Parameter, S1, "pearson_q_BH")
  key$q_2025_26 <- mapply(g, key$Target, key$Parameter, S2, "pearson_q_BH")
  key$same_sign <- sign(key$r_2024_25) == sign(key$r_2025_26)
  key$both_sig_BH05 <- key$q_2024_25 < 0.05 & key$q_2025_26 < 0.05
  key$mean_abs_r <- (abs(key$r_2024_25) + abs(key$r_2025_26)) / 2
  key$consistent <- key$same_sign & key$both_sig_BH05
  key$consistent[is.na(key$consistent)] <- FALSE
  key[order(-key$mean_abs_r), ]
}

# ---------------------------------------------------------------------------
# 2. Type II three-way ANOVA: water regime x nitrogen x seaweed
#
# A term is tested against the model containing every term that does NOT
# contain it. Testing a main effect while its own interactions stay in the
# model would make that test nearly meaningless.
# ---------------------------------------------------------------------------
TERM_FACTORS <- list(Water = "W", N = "N", Seaweed = "S",
                     WxN = c("W","N"), WxS = c("W","S"), NxS = c("N","S"),
                     WxNxS = c("W","N","S"))
TERM_ORDER <- names(TERM_FACTORS)

anova3 <- function(df, params) {
  rows <- list(); k <- 0L
  for (season in c(S1, S2)) {
    d0 <- df[df$Season == season, ]
    W0 <- model.matrix(~ Water_Regime, d0)[, -1, drop = FALSE]
    N0 <- model.matrix(~ N_Level, d0)[, -1, drop = FALSE]
    S0 <- model.matrix(~ Seaweed_Level, d0)[, -1, drop = FALSE]
    inter <- function(A, B) {
      out <- NULL
      for (i in seq_len(ncol(A))) for (j in seq_len(ncol(B)))
        out <- cbind(out, A[, i] * B[, j])
      out
    }
    blk <- list(Water = W0, N = N0, Seaweed = S0,
                WxN = inter(W0, N0), WxS = inter(W0, S0), NxS = inter(N0, S0),
                WxNxS = inter(inter(W0, N0), S0))
    for (p in params) {
      y_all <- suppressWarnings(as.numeric(d0[[p]]))
      keep <- is.finite(y_all)
      if (sum(keep) < 40 || sd(y_all[keep]) == 0) next
      y <- y_all[keep]
      B <- lapply(blk, function(M) M[keep, , drop = FALSE])
      rss <- function(M) {
        A <- cbind(1, M)
        fit <- qr(A)
        res <- qr.resid(fit, y)
        c(sum(res^2), fit$rank)
      }
      full <- do.call(cbind, B[TERM_ORDER])
      rf <- rss(full); rss_full <- rf[1]; k_full <- rf[2]
      dfe <- length(y) - k_full
      tss <- sum((y - mean(y))^2)
      rec <- list(Season = season, Parameter = p, n = length(y), df_resid = dfe,
                  R2_model = if (tss > 0) 1 - rss_full / tss else NA_real_)
      if (dfe > 0 && rss_full > 0) {
        mse <- rss_full / dfe
        for (t in TERM_ORDER) {
          keep_terms <- TERM_ORDER[!vapply(TERM_ORDER, function(o)
            all(TERM_FACTORS[[t]] %in% TERM_FACTORS[[o]]), logical(1))]
          red <- if (length(keep_terms)) do.call(cbind, B[keep_terms]) else NULL
          add <- cbind(red, B[[t]])
          r1 <- rss(red %||% matrix(nrow = length(y), ncol = 0))
          r2 <- rss(add)
          q <- r2[2] - r1[2]
          if (q <= 0) next
          F <- ((r1[1] - r2[1]) / q) / mse
          rec[[paste0("F_", t)]] <- F
          rec[[paste0("p_", t)]] <- pf(F, q, dfe, lower.tail = FALSE)
        }
      }
      k <- k + 1L
      rows[[k]] <- as.data.frame(rec, stringsAsFactors = FALSE)
    }
  }
  cols <- unique(unlist(lapply(rows, names)))
  out <- do.call(rbind, lapply(rows, function(r) {
    for (c0 in setdiff(cols, names(r))) r[[c0]] <- NA_real_
    r[, cols, drop = FALSE]
  }))
  for (t in TERM_ORDER) {
    pc <- paste0("p_", t)
    if (pc %in% names(out)) {
      out[[paste0("q_", t)]] <- NA_real_
      for (season in c(S1, S2)) {
        m <- out$Season == season & !is.na(out[[pc]])
        if (any(m)) out[[paste0("q_", t)]][m] <- p.adjust(out[[pc]][m], "BH")
      }
    }
  }
  out
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# ---------------------------------------------------------------------------
# 3. season comparison: Welch t and Hedges' g
# ---------------------------------------------------------------------------
season_compare <- function(df, params) {
  rows <- list(); k <- 0L
  for (p in params) {
    a <- suppressWarnings(as.numeric(df[[p]][df$Season == S1])); a <- a[is.finite(a)]
    b <- suppressWarnings(as.numeric(df[[p]][df$Season == S2])); b <- b[is.finite(b)]
    if (length(a) < 5 || length(b) < 5) next
    tt <- tryCatch(t.test(a, b, var.equal = FALSE), error = function(e) NULL)
    if (is.null(tt)) next
    sp <- sqrt(((length(a) - 1) * var(a) + (length(b) - 1) * var(b)) /
               (length(a) + length(b) - 2))
    g <- if (sp > 0) (mean(a) - mean(b)) / sp else NA_real_
    g <- g * (1 - 3 / (4 * (length(a) + length(b)) - 9))
    k <- k + 1L
    rows[[k]] <- data.frame(Parameter = p, mean_2024_25 = mean(a), sd_2024_25 = sd(a),
                            mean_2025_26 = mean(b), sd_2025_26 = sd(b),
                            welch_t = unname(tt$statistic), p = tt$p.value,
                            hedges_g = g, stringsAsFactors = FALSE)
  }
  out <- do.call(rbind, rows)
  out$q_BH <- p.adjust(out$p, "BH")
  out
}

# ---------------------------------------------------------------------------
main <- function() {
  bm <- build_matrix(); df <- bm$df; blocks <- bm$blocks
  params <- names(df)[vapply(df, is.numeric, logical(1))]
  params <- setdiff(params, c("Season_ID", "N_kg_ha", "Seaweed_pct"))
  params <- params[vapply(params, function(p)
    sum(!is.na(df[[p]])) > 20 && length(unique(df[[p]][!is.na(df[[p]])])) > 1, logical(1))]
  cat("screening", length(params), "parameters\n")

  corr <- correlations(df, blocks, params); write_tbl(corr, "univariate_correlations")
  cons <- consistency(corr);                write_tbl(cons, "univariate_consistency")
  write_tbl(cons[cons$Target == "Grain_Yield_t_ha", ], "univariate_top")

  av <- anova3(df, params);                 write_tbl(av, "treatment_effects")
  sc <- season_compare(df, params);         write_tbl(sc, "season_comparison")

  cat("\nTOP 15 parameters vs GRAIN YIELD (consistent in both seasons)\n")
  t <- cons[cons$Target == "Grain_Yield_t_ha" & cons$consistent, ]
  print(head(t[, c("Parameter","Block","r_2024_25","r_2025_26","mean_abs_r")], 15),
        row.names = FALSE, digits = 3)

  cat("\nconsistent associations by block (grain yield):\n")
  print(sort(table(t$Block), decreasing = TRUE))

  cat("\nTREATMENT EFFECTS: parameters with q < 0.05, by term and season\n")
  qc <- grep("^q_", names(av), value = TRUE)
  print(do.call(rbind, lapply(split(av, av$Season), function(g)
    vapply(qc, function(c0) sum(g[[c0]] < 0.05, na.rm = TRUE), numeric(1)))))

  cat("\nresults ->", RESD, "\n")
  invisible(list(df = df, blocks = blocks, corr = corr, cons = cons, av = av, sc = sc))
}

res <- main()
