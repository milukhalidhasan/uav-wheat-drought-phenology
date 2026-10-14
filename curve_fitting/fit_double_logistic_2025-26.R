###############################################################################
#  VI PHENOLOGY ANALYSIS - wheat UAV trial
#
#  Design : 2 water regimes (D = drought, I = irrigated, main plot)
#           x 12 treatments (T0-T11, subplot) x 4 replications = 96 plots
#           10 UAV flights, 19-83 DAS; harvest 96 DAS
#
#  PART 1  data
#  PART 2  double-logistic model and scale-free fitter
#  PART 3  redundancy audit across every spectral layer
#  PART 4  screening: fit all layers, score them, apply a selection rule
#  PART 5  panel parameters (per plot) for the retained indices
#  PART 6  split-plot ANOVA and D vs I contrasts on phenology parameters
#  PART 7  yield models with leave-one-out cross-validation
#  PART 8  figures
#
#  Model:  y(t) = y0 + (y1 - y0) * [ 1/(1+exp(-k1(t-t1))) - 1/(1+exp(-k2(t-t2))) ]
#     y0 background value    y1 plateau value
#     t1 green-up midpoint   t2 senescence midpoint
#     k1 green-up rate       k2 senescence rate
###############################################################################

suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(tidyr); library(purrr)
  library(ggplot2); library(minpack.lm)
})

# season workbook; not included in the repository, see README
in_file <- Sys.getenv("WHEAT_S2_XLSX", unset = "MegaData2025_Cleaned.xlsx")
OUT     <- "outputs"
FIG     <- file.path(OUT, "figures")
dir.create(FIG, showWarnings = FALSE, recursive = TRUE)

set.seed(1)

## ===========================================================  PART 1  data ===
spec <- read_excel(in_file, sheet = "Spectral_TimeSeries")
harv <- read_excel(in_file, sheet = "Harvest_PlotLevel")

trt_levels <- paste0("T", 0:11)

meta <- spec %>%
  distinct(Plot_ID, Water_Regime, Replication, Treatment) %>%
  left_join(harv %>% select(Plot_ID, GY = `Grain_Yield(t/ha)`), by = "Plot_ID") %>%
  mutate(Treatment    = factor(Treatment, levels = trt_levels),
         Replication  = factor(Replication),
         Water_Regime = factor(Water_Regime, levels = c("D", "I")))

# every spectral layer in the workbook, plus thermal and canopy height
vi_cols <- names(spec)[which(names(spec) == "MGVRI"):which(names(spec) == "RDVI")]
cand    <- c(vi_cols, "Thermal", "dchm")

wr_lab <- c(D = "Drought", I = "Irrigated")
wr_col <- c(D = "#d1495b", I = "#1f6fb4")

## ==============================================  PART 2  model and fitter ===
dbl_log <- function(t, y0, y1, t1, t2, k1, k2)
  y0 + (y1 - y0) * (1 / (1 + exp(-k1 * (t - t1))) - 1 / (1 + exp(-k2 * (t - t2))))

# Scale-free fitter: bounds are set from each series' own range, so one routine
# serves NDVI (0-1), GCI (0-8) and RVI (2-20) without retuning. Time bounds are
# tied to the observed flight window rather than hard-coded, because this trial
# ends at 83 DAS while the earlier one ran to 92.
fit_generic <- function(t, y) {
  ok <- is.finite(t) & is.finite(y); t <- t[ok]; y <- y[ok]
  if (length(t) < 7 || diff(range(y)) == 0) return(NULL)
  lo <- min(y); hi <- max(y); rg <- hi - lo
  tmin <- min(t); tmax <- max(t); tr <- tmax - tmin

  LB <- c(lo - 1.0 * rg, lo + 0.15 * rg, tmin - 15,        tmin + 0.40 * tr, 0.01, 0.01)
  UB <- c(lo + 0.6 * rg, hi + 1.5 * rg,  tmin + 0.7 * tr,  tmax + 25,        0.80, 0.80)
  st <- pmin(pmax(c(lo, hi, tmin + 0.30 * tr, tmin + 0.85 * tr, 0.15, 0.20),
                  LB + 1e-9), UB - 1e-9)
  names(st) <- names(LB) <- names(UB) <- c("y0","y1","t1","t2","k1","k2")

  f <- tryCatch(nlsLM(y ~ dbl_log(t, y0, y1, t1, t2, k1, k2),
                      data = data.frame(t = t, y = y), start = as.list(st),
                      lower = LB, upper = UB,
                      control = nls.lm.control(maxiter = 1024, ftol = 1e-10)),
                error = function(e) NULL)
  if (is.null(f)) return(NULL)

  p  <- coef(f); pr <- predict(f)
  hit <- sum(abs(p - LB) < 1e-3 * pmax(1, abs(LB)) |
             abs(p - UB) < 1e-3 * pmax(1, abs(UB)))
  g  <- seq(min(t), max(t), by = 0.5)
  fv <- dbl_log(g, p[1], p[2], p[3], p[4], p[5], p[6])

  tibble(y0 = p[1], y1 = p[2], t1 = p[3], t2 = p[4], k1 = p[5], k2 = p[6],
         Amplitude = p[2] - p[1], Green_Duration = p[4] - p[3],
         AUC  = sum(diff(g) * (head(fv, -1) + tail(fv, -1)) / 2),
         Peak = max(fv),
         R2   = 1 - sum((y - pr)^2) / sum((y - mean(y))^2),
         RMSE = sqrt(mean((y - pr)^2)),
         at_bound = hit)
}

# fit one layer for every plot
fit_index <- function(vi) {
  spec %>% group_split(Plot_ID) %>%
    map_dfr(function(d) {
      d <- arrange(d, DAS)
      r <- fit_generic(d$DAS, d[[vi]])
      if (is.null(r)) return(NULL)
      mutate(r, Plot_ID = d$Plot_ID[1])
    }) %>%
    left_join(meta, by = "Plot_ID")
}

# exact leave-one-out CV for a linear model, via the hat matrix
# (identical to caret::train(method = "lm", trControl = trainControl("LOOCV"))
#  but with no dependency and no refitting loop)
loocv <- function(form, data) {
  data <- na.omit(data[, all.vars(form), drop = FALSE])
  if (nrow(data) < length(all.vars(form)) + 3) return(c(R2 = NA, RMSE = NA))
  m  <- lm(form, data)
  h  <- lm.influence(m)$hat
  e  <- residuals(m) / (1 - h)              # PRESS residuals
  y  <- data[[all.vars(form)[1]]]
  c(R2 = unname(cor(y - e, y)^2), RMSE = sqrt(mean(e^2)))
}

## ==========================================  PART 3  redundancy audit ===
# Layers that are algebraic rearrangements of each other carry one result, not
# several. Two thresholds: r > 0.9995 flags outright duplicates, r > 0.99 flags
# near-duplicates that should not both appear in a results table.
pairs_r <- combn(cand, 2, simplify = FALSE) %>%
  map_dfr(function(p) {
    r <- suppressWarnings(cor(spec[[p[1]]], spec[[p[2]]], use = "complete.obs"))
    if (is.na(r)) return(NULL)
    tibble(a = p[1], b = p[2], r = r)
  })

dupes <- pairs_r %>% filter(abs(r) > 0.9995) %>%
  mutate(identical = map2_lgl(a, b, ~ isTRUE(all.equal(spec[[.x]], spec[[.y]]))))
near  <- pairs_r %>% filter(abs(r) > 0.99, abs(r) <= 0.9995) %>% arrange(desc(abs(r)))

cat("\n-- exact duplicates (|r| > 0.9995) --\n"); print(as.data.frame(dupes))
cat("\n-- near duplicates (|r| > 0.99) --\n");     print(as.data.frame(near))
write.csv(pairs_r, file.path(OUT, "index_pairwise_correlations.csv"), row.names = FALSE)

# data-driven families: cluster layers on |r| and cut at 0.95
cm <- spec %>% select(all_of(cand)) %>% cor(use = "pairwise.complete.obs")
cm[is.na(cm)] <- 0
families <- tibble(VI = colnames(cm),
                   Family = cutree(hclust(as.dist(1 - abs(cm)), "average"), h = 0.05))
write.csv(families, file.path(OUT, "index_families.csv"), row.names = FALSE)

## ======================================================  PART 4  screening ===
# Score every layer on: how well the double logistic fits it, how often the
# optimiser lands on a bound (a fit that hits a bound is a failed fit wearing a
# high R2), whether t2 is biologically plausible, how strongly it separates the
# two water regimes, and how well its parameters predict grain yield.
#
# t2 plausibility is judged against this trial: heading was around 60 DAS and
# harvest at 96 DAS, so a senescence midpoint outside 65-100 DAS is rejected.
T2_LO <- 65; T2_HI <- 100

score_one <- function(vi) {
  P <- fit_index(vi)
  if (nrow(P) < 60) return(tibble(VI = vi, converged = nrow(P)))
  a <- P$t2[P$Water_Regime == "I"]; b <- P$t2[P$Water_Regime == "D"]

  c6  <- loocv(GY ~ y0 + y1 + t1 + t2 + k1 + k2, P)
  ca  <- loocv(GY ~ AUC,  P)
  cpk <- loocv(GY ~ Peak, P)                      # single-date-style baseline
  # within-regime CV: pooled CV is inflated because drought and irrigated
  # differ in both the index and the yield, so a regime marker alone scores well
  cD  <- loocv(GY ~ AUC, filter(P, Water_Regime == "D"))
  cI  <- loocv(GY ~ AUC, filter(P, Water_Regime == "I"))

  tibble(VI = vi, converged = nrow(P),
         R2_mean = mean(P$R2), R2_min = min(P$R2),
         pct_at_bound   = 100 * mean(P$at_bound > 0),
         t2_mean        = mean(P$t2),
         t2_implausible = 100 * mean(P$t2 < T2_LO | P$t2 > T2_HI),
         cohen_d = (mean(a) - mean(b)) / sqrt((var(a) + var(b)) / 2),
         t2_p    = wilcox.test(a, b, exact = FALSE)$p.value,
         LOOCV_6par = c6["R2"], LOOCV_AUC = ca["R2"], LOOCV_Peak = cpk["R2"],
         LOOCV_AUC_D = cD["R2"], LOOCV_AUC_I = cI["R2"],
         RMSE_AUC = ca["RMSE"])
}

message("screening ", length(cand), " layers x 96 plots ...")
screen <- map_dfr(cand, score_one) %>% arrange(desc(R2_mean))
screen$t2_p_BH <- p.adjust(screen$t2_p, "BH")
write.csv(screen, file.path(OUT, "index_screening_all_layers.csv"), row.names = FALSE)
print(as.data.frame(screen %>% mutate(across(where(is.numeric), ~ round(.x, 3)))))

## selection rule ------------------------------------------------------------
# Keep a layer only if it clears all four: the curve fits, the fit is honest
# (few plots at a bound), the senescence date is plausible, and the seasonal
# integral carries yield signal.
selected <- screen %>%
  filter(R2_mean > 0.97, pct_at_bound < 30, t2_implausible < 10, LOOCV_AUC > 0.55) %>%
  pull(VI)
message("Selected: ", paste(selected, collapse = ", "))

# One index per correlation family, so a family is not reported several times.
PANEL <- screen %>%
  filter(VI %in% selected) %>%
  left_join(families, by = "VI") %>%
  group_by(Family) %>% slice_max(R2_mean, n = 1) %>% ungroup() %>%
  arrange(desc(R2_mean)) %>% pull(VI)
message("Panel after family reduction: ", paste(PANEL, collapse = ", "))

# Reported panel: the family representatives above, plus NDVI and gNDVI kept for
# continuity with the literature even where they do not clear every criterion.
REPORT <- unique(c(PANEL, "NDVI", "gNDVI", "NDRE", "CI(rededge)", "GCI", "RVI"))
REPORT <- REPORT[REPORT %in% screen$VI[screen$converged >= 60]]

## ===============================================  PART 5  panel parameters ===
params <- map_dfr(REPORT, ~ mutate(fit_index(.x), Index = .x))
write.csv(params, file.path(OUT, "panel_parameters_plotlevel.csv"), row.names = FALSE)

## ============================  PART 6  split-plot ANOVA and D vs I tests ===
# The trial is a split plot: water regime on main plots, treatment on subplots.
# Testing the water-regime effect against the residual would use the wrong error
# term, so the model carries Error(Replication/Water_Regime).
splitplot_p <- function(d) {
  m <- try(aov(Value ~ Water_Regime * Treatment + Error(Replication / Water_Regime),
               data = d), silent = TRUE)
  out <- c(p_regime = NA_real_, p_treatment = NA_real_, p_interaction = NA_real_)
  if (inherits(m, "try-error")) return(out)
  s <- summary(m)
  main <- s[[grep("Replication:Water_Regime", names(s))[1]]][[1]]
  within <- s[[grep("Within", names(s))[1]]][[1]]
  gp <- function(tab, key) {
    i <- grep(key, trimws(rownames(tab)))
    if (length(i) == 0) NA_real_ else tab[i[1], "Pr(>F)"]
  }
  out["p_regime"]      <- gp(main,   "^Water_Regime$")
  out["p_treatment"]   <- gp(within, "^Treatment$")
  out["p_interaction"] <- gp(within, "Water_Regime:Treatment")
  out
}

tests <- params %>%
  pivot_longer(c(y0, y1, t1, t2, k1, k2, Amplitude, Green_Duration, AUC),
               names_to = "Parameter", values_to = "Value") %>%
  group_by(Index, Parameter) %>%
  group_modify(function(d, key) {
    vI <- d$Value[d$Water_Regime == "I"]; vD <- d$Value[d$Water_Regime == "D"]
    pp <- splitplot_p(d)
    tibble(I = mean(vI), D = mean(vD), diff = mean(vI) - mean(vD),
           cohen_d = (mean(vI) - mean(vD)) / sqrt((var(vI) + var(vD)) / 2),
           p_wilcox = wilcox.test(Value ~ Water_Regime, data = d, exact = FALSE)$p.value,
           p_regime = pp["p_regime"], p_treatment = pp["p_treatment"],
           p_interaction = pp["p_interaction"])
  }) %>%
  ungroup() %>%
  group_by(Parameter) %>%
  mutate(p_wilcox_BH = p.adjust(p_wilcox, "BH")) %>%
  ungroup()

write.csv(tests, file.path(OUT, "phenology_parameter_tests.csv"), row.names = FALSE)
cat("\n-- senescence midpoint (t2), ranked by effect size --\n")
print(as.data.frame(tests %>% filter(Parameter == "t2") %>% arrange(desc(abs(cohen_d))) %>%
                    mutate(across(where(is.numeric), ~ signif(.x, 3)))))

## reference test: the same split-plot model applied to grain yield itself, so
## the phenology results can be read against what the agronomic data supports
yield_aov <- aov(GY ~ Water_Regime * Treatment + Error(Replication / Water_Regime),
                 data = meta)
cat("\n-- grain yield, split-plot ANOVA --\n"); print(summary(yield_aov))
capture.output(summary(yield_aov), file = file.path(OUT, "grain_yield_anova.txt"))

## =========================================  PART 7  yield models with LOOCV ===
wide <- params %>%
  select(Plot_ID, Index, y0:AUC, Peak) %>%
  pivot_wider(names_from = Index,
              values_from = c(y0, y1, t1, t2, k1, k2, Amplitude, Green_Duration, AUC, Peak),
              names_sep = "_") %>%
  left_join(meta, by = "Plot_ID") %>% na.omit()
names(wide) <- make.names(names(wide))

cv_row <- function(vi, lab, r)
  tibble(Index = vi, Model = lab, R2 = unname(r["R2"]), RMSE = unname(r["RMSE"]))

perf <- map_dfr(REPORT, function(vi) {
  v  <- make.names(vi)
  f6 <- as.formula(sprintf("GY ~ y0_%s + y1_%s + t1_%s + t2_%s + k1_%s + k2_%s",
                           v, v, v, v, v, v))
  fa <- as.formula(sprintf("GY ~ AUC_%s", v))
  fr <- as.formula(sprintf("GY ~ AUC_%s + Water_Regime", v))
  fp <- as.formula(sprintf("GY ~ Peak_%s", v))
  bind_rows(
    cv_row(vi, "6 parameters",        loocv(f6, wide)),
    cv_row(vi, "AUC alone",           loocv(fa, wide)),
    cv_row(vi, "AUC + regime",        loocv(fr, wide)),
    cv_row(vi, "Peak value alone",    loocv(fp, wide)),
    cv_row(vi, "AUC, drought only",   loocv(fa, filter(wide, Water_Regime == "D"))),
    cv_row(vi, "AUC, irrigated only", loocv(fa, filter(wide, Water_Regime == "I")))
  )
}) %>% arrange(desc(R2))

write.csv(perf, file.path(OUT, "yield_models_loocv.csv"), row.names = FALSE)
cat("\n-- yield prediction, leave-one-out CV --\n")
print(as.data.frame(perf %>% mutate(across(where(is.numeric), ~ round(.x, 3)))), row.names = FALSE)

## =============================================================  PART 8  figures ===
theme_pheno <- theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(linewidth = 0.25, colour = "grey90"),
        strip.background = element_rect(fill = "grey92", colour = "grey40"),
        strip.text       = element_text(face = "bold"),
        legend.position  = "bottom",
        legend.title     = element_text(face = "bold"))

pal12 <- c("#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c",
           "#fdbf6f","#ff7f00","#cab2d6","#6a3d9a","#c9b458","#b15928")
names(pal12) <- trt_levels

## ---- 8.1 group-level curves per water regime x treatment -------------------
# Two-stage fit: plot-level fits above supply the starting values for a single
# curve per water regime x treatment, fitted to all four replicates pooled.
long <- spec %>%
  select(Plot_ID, Water_Regime, Replication, Treatment, DAS, all_of(REPORT)) %>%
  pivot_longer(all_of(REPORT), names_to = "Index", values_to = "Value") %>%
  filter(is.finite(Value)) %>%
  mutate(Treatment = factor(Treatment, levels = trt_levels),
         Water_Regime = factor(Water_Regime, levels = c("D", "I")))

raw_summ <- long %>%
  group_by(Index, Water_Regime, Treatment, DAS) %>%
  summarise(mean = mean(Value), sd = sd(Value), se = sd(Value) / sqrt(n()),
            .groups = "drop")

starts <- params %>%
  group_by(Index, Water_Regime, Treatment) %>%
  summarise(across(c(y0, y1, t1, t2, k1, k2), mean), .groups = "drop")

fit_group <- function(df, start) {
  t <- df$DAS; y <- df$Value
  lo <- min(y); hi <- max(y); rg <- hi - lo
  tmin <- min(t); tmax <- max(t); tr <- tmax - tmin
  LB <- c(y0 = lo - 1.0*rg, y1 = lo + 0.15*rg, t1 = tmin - 15,
          t2 = tmin + 0.40*tr, k1 = 0.01, k2 = 0.01)
  UB <- c(y0 = lo + 0.6*rg, y1 = hi + 1.5*rg, t1 = tmin + 0.7*tr,
          t2 = tmax + 25, k1 = 0.80, k2 = 0.80)
  st <- as.list(pmin(pmax(unlist(start)[names(LB)], LB + 1e-9), UB - 1e-9))
  f <- tryCatch(nlsLM(Value ~ dbl_log(DAS, y0, y1, t1, t2, k1, k2), data = df,
                      start = st, lower = LB, upper = UB,
                      control = nls.lm.control(maxiter = 1024, maxfev = 10000)),
                error = function(e) NULL)
  f
}

grp <- map_dfr(seq_len(nrow(starts)), function(i) {
  g  <- starts[i, ]
  df <- long %>% filter(Index == g$Index, Water_Regime == g$Water_Regime,
                        Treatment == g$Treatment)
  f <- fit_group(df, g[, c("y0","y1","t1","t2","k1","k2")])
  if (is.null(f)) return(NULL)
  p  <- as.list(coef(f))
  mn <- raw_summ %>% filter(Index == g$Index, Water_Regime == g$Water_Regime,
                            Treatment == g$Treatment)
  gr <- data.frame(DAS = seq(min(df$DAS), max(df$DAS), length.out = 400))
  gr$Value <- with(p, dbl_log(gr$DAS, y0, y1, t1, t2, k1, k2))
  gr$Index <- g$Index; gr$Water_Regime <- g$Water_Regime; gr$Treatment <- g$Treatment
  gr$R2_mean <- 1 - sum((mn$mean - with(p, dbl_log(mn$DAS, y0,y1,t1,t2,k1,k2)))^2) /
                    sum((mn$mean - mean(mn$mean))^2)
  gr
})

group_pars <- map_dfr(seq_len(nrow(starts)), function(i) {
  g  <- starts[i, ]
  df <- long %>% filter(Index == g$Index, Water_Regime == g$Water_Regime,
                        Treatment == g$Treatment)
  f <- fit_group(df, g[, c("y0","y1","t1","t2","k1","k2")]); if (is.null(f)) return(NULL)
  p <- as.list(coef(f))
  gg <- seq(min(df$DAS), max(df$DAS), by = 0.5)
  fv <- with(p, dbl_log(gg, y0, y1, t1, t2, k1, k2))
  as_tibble(p) %>% mutate(Index = g$Index, Water_Regime = g$Water_Regime,
                          Treatment = g$Treatment,
                          Amplitude = y1 - y0, Green_Duration = t2 - t1,
                          AUC = sum(diff(gg) * (head(fv,-1) + tail(fv,-1)) / 2))
})
write.csv(group_pars, file.path(OUT, "double_logistic_parameters_by_treatment.csv"),
          row.names = FALSE)

panel_lev <- c("(a) Raw", "(b) Double logistic fit")

fig_main <- function(idx) {
  a <- raw_summ %>% filter(Index == idx) %>% mutate(Panel = factor(panel_lev[1], panel_lev))
  b <- raw_summ %>% filter(Index == idx) %>% mutate(Panel = factor(panel_lev[2], panel_lev))
  c <- grp      %>% filter(Index == idx) %>% mutate(Panel = factor(panel_lev[2], panel_lev))
  ggplot() +
    geom_errorbar(data = a, aes(DAS, ymin = mean - sd, ymax = mean + sd, colour = Treatment),
                  width = 1.2, linewidth = 0.3, alpha = 0.5,
                  position = position_dodge(width = 1.8), show.legend = FALSE) +
    geom_line(data = a, aes(DAS, mean, colour = Treatment), linewidth = 0.45) +
    geom_point(data = a, aes(DAS, mean, colour = Treatment), size = 1.1) +
    geom_line(data = c, aes(DAS, Value, colour = Treatment), linewidth = 0.65) +
    geom_point(data = b, aes(DAS, mean, colour = Treatment), size = 1.1) +
    facet_grid(Panel ~ Water_Regime, switch = "y",
               labeller = labeller(Water_Regime = wr_lab)) +
    scale_colour_manual(values = pal12, name = "Treatment") +
    scale_x_continuous(breaks = seq(20, 80, 20)) +
    labs(x = "Days After Sowing (DAS)", y = paste("Mean", idx)) +
    guides(colour = guide_legend(nrow = 1, override.aes = list(linewidth = 1))) +
    theme_pheno
}

fig_bytrt <- function(idx) {
  a <- raw_summ %>% filter(Index == idx); c <- grp %>% filter(Index == idx)
  ggplot() +
    geom_errorbar(data = a, aes(DAS, ymin = mean - sd, ymax = mean + sd, colour = Water_Regime),
                  width = 1.2, linewidth = 0.3, alpha = 0.6, show.legend = FALSE) +
    geom_point(data = a, aes(DAS, mean, colour = Water_Regime), size = 1.2) +
    geom_line(data = c, aes(DAS, Value, colour = Water_Regime), linewidth = 0.7) +
    facet_wrap(~ Treatment, ncol = 4) +
    scale_colour_manual(values = wr_col, labels = wr_lab, name = "Water regime") +
    scale_x_continuous(breaks = seq(20, 80, 20)) +
    labs(x = "Days After Sowing (DAS)", y = paste("Mean", idx)) +
    theme_pheno
}

for (idx in REPORT) {
  safe <- gsub("[^A-Za-z0-9]", "", idx)
  ggsave(file.path(FIG, paste0("Fig_phenology_", safe, ".png")), fig_main(idx),
         width = 9, height = 6.6, dpi = 400)
  ggsave(file.path(FIG, paste0("Fig_phenology_", safe, ".pdf")), fig_main(idx),
         width = 9, height = 6.6)
  ggsave(file.path(FIG, paste0("FigS_bytreatment_", safe, ".png")), fig_bytrt(idx),
         width = 10, height = 7, dpi = 400)
}

## ---- 8.2 multi-index panel, water regimes contrasted -----------------------
curves_panel <- grp %>%
  group_by(Index, Water_Regime, DAS) %>%
  summarise(fit = mean(Value), .groups = "drop")
obs_panel <- long %>%
  group_by(Index, Water_Regime, DAS) %>%
  summarise(m = mean(Value), se = sd(Value) / sqrt(n()), .groups = "drop")

p_panel <- ggplot() +
  geom_errorbar(data = obs_panel, aes(DAS, ymin = m - se, ymax = m + se,
                                      colour = Water_Regime), width = 1) +
  geom_point(data = obs_panel, aes(DAS, m, colour = Water_Regime), size = 1.3) +
  geom_line(data = curves_panel, aes(DAS, fit, colour = Water_Regime), linewidth = 0.8) +
  facet_wrap(~ Index, scales = "free_y", ncol = 3) +
  scale_colour_manual(values = wr_col, labels = wr_lab, name = NULL) +
  labs(x = "Days after sowing", y = "Index value") +
  theme_pheno
ggsave(file.path(FIG, "Fig_multiindex_curves.png"), p_panel, width = 8.5, height = 6, dpi = 400)
ggsave(file.path(FIG, "Fig_multiindex_curves.pdf"), p_panel, width = 8.5, height = 6)

## ---- 8.3 screening map -----------------------------------------------------
sc <- screen %>% filter(converged >= 60)
p_screen <- ggplot(sc, aes(pct_at_bound, R2_mean)) +
  geom_rect(aes(xmin = -Inf, xmax = 30, ymin = 0.97, ymax = Inf),
            fill = "#e8f2e8", colour = NA, inherit.aes = FALSE) +
  geom_point(aes(size = LOOCV_AUC, colour = t2_implausible < 10), alpha = 0.85) +
  ggrepel::geom_text_repel(aes(label = VI), size = 2.6, max.overlaps = 20,
                           segment.size = 0.2) +
  scale_colour_manual(values = c(`TRUE` = "#1f6fb4", `FALSE` = "#d1495b"),
                      labels = c(`TRUE` = "plausible", `FALSE` = "implausible"),
                      name = "senescence date") +
  scale_size_continuous(name = expression(LOOCV~R^2~"(AUC)"), range = c(1, 5)) +
  labs(x = "Plots with a parameter at a bound (%)",
       y = expression(mean~fit~R^2),
       title = "Double-logistic suitability of every spectral layer",
       subtitle = "shaded area: passes the fit and bound criteria") +
  theme_pheno
ggsave(file.path(FIG, "Fig_index_screening.png"), p_screen, width = 8, height = 6, dpi = 400)

## ---- 8.4 phenology parameters by treatment and regime ----------------------
p_par <- params %>%
  filter(Index %in% REPORT[1:min(6, length(REPORT))]) %>%
  # only parameters in days or rate units, so a row shares a comparable scale
  # across indices; AUC and the amplitude terms are index-scaled and live in the
  # parameter CSV instead
  pivot_longer(c(t1, t2, Green_Duration, k2), names_to = "Parameter", values_to = "Value") %>%
  ggplot(aes(Treatment, Value, colour = Water_Regime)) +
  stat_summary(fun.data = mean_se, position = position_dodge(0.5), size = 0.3) +
  facet_grid(Parameter ~ Index, scales = "free_y") +
  scale_colour_manual(values = wr_col, labels = wr_lab, name = "Water regime") +
  labs(x = "Treatment", y = "Parameter value (mean +/- SE)") +
  theme_pheno +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, size = 6))
ggsave(file.path(FIG, "Fig_parameters_by_treatment.png"), p_par,
       width = 11, height = 7.5, dpi = 400)

## ---- 8.5 observed vs LOOCV-predicted yield ---------------------------------
best <- perf %>% filter(Model == "AUC alone") %>% slice_max(R2, n = 1) %>% pull(Index)
v    <- make.names(best)
fa   <- as.formula(sprintf("GY ~ AUC_%s", v))
mm   <- lm(fa, wide); hh <- lm.influence(mm)$hat
pred <- wide$GY - residuals(mm) / (1 - hh)

p_yield <- ggplot(data.frame(obs = wide$GY, pred = pred,
                             Water_Regime = wide$Water_Regime),
                  aes(pred, obs, colour = Water_Regime)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "grey50") +
  geom_point(size = 1.8, alpha = 0.85) +
  scale_colour_manual(values = wr_col, labels = wr_lab, name = "Water regime") +
  coord_equal() +
  labs(x = "LOOCV predicted grain yield (t/ha)", y = "Observed grain yield (t/ha)",
       title = paste0("Yield from seasonal ", best, " integral (AUC)")) +
  theme_pheno
ggsave(file.path(FIG, "Fig_yield_loocv.png"), p_yield, width = 6, height = 5.5, dpi = 400)

## =====================================  PART 9  phenological stage values ===
# Every stage below is derived analytically from the fitted double logistic, so
# a stage is a property of the curve rather than of the flight dates.
#
#   SOS   start of season          t1 - ln(9)/k1      (10% of the rise)
#   t1    green-up inflection      parameter          (fastest green-up)
#   EOG   end of green-up          t1 + ln(9)/k1      (90% of the rise)
#   POS   peak of season           argmax of the fitted curve
#   SOD   start of decline         t2 - ln(9)/k2      (10% of the fall)
#   t2    senescence inflection    parameter          (fastest senescence)
#   EOS   end of season            t2 + ln(9)/k2      (90% of the fall)
#
#   Rate_greenup    = A*k1/4   index units per day at t1
#   Rate_senescence = A*k2/4   index units per day at t2
#   Green_Duration  = t2 - t1        Plateau = SOD - EOG
#   LOS             = EOS - SOS      Peak_value = curve maximum
#
# Stage dates after the last flight (83 DAS) are extrapolations of the model,
# not observations; the `n_extrapolated` column counts them.
L9        <- log(9)
LAST_FLY  <- max(spec$DAS)
FIRST_FLY <- min(spec$DAS)

stage_from_par <- function(p) {
  A  <- p$y1 - p$y0
  gg <- seq(0, 140, by = 0.05)
  fv <- dbl_log(gg, p$y0, p$y1, p$t1, p$t2, p$k1, p$k2)
  tibble(
    SOS = p$t1 - L9 / p$k1,
    t1  = p$t1,
    EOG = p$t1 + L9 / p$k1,
    POS = gg[which.max(fv)],
    SOD = p$t2 - L9 / p$k2,
    t2  = p$t2,
    EOS = p$t2 + L9 / p$k2,
    Peak_value      = max(fv),
    Rate_greenup    = A * p$k1 / 4,
    Rate_senescence = A * p$k2 / 4,
    Green_Duration  = p$t2 - p$t1,
    Plateau         = (p$t2 - L9 / p$k2) - (p$t1 + L9 / p$k1),
    LOS             = (p$t2 + L9 / p$k2) - (p$t1 - L9 / p$k1)
  )
}

stage_cols <- c("SOS","t1","EOG","POS","SOD","t2","EOS",
                "Peak_value","Rate_greenup","Rate_senescence",
                "Green_Duration","Plateau","LOS")

## 9.1 stage values for every plot ------------------------------------------
stages_plot <- params %>%
  group_by(Index, Plot_ID, Water_Regime, Replication, Treatment, GY) %>%
  group_modify(function(d, k) stage_from_par(as.list(d[1, ]))) %>%
  ungroup() %>%
  mutate(n_outside_window =
           rowSums(across(c(SOS, t1, EOG, POS, SOD, t2, EOS),
                          ~ .x < FIRST_FLY | .x > LAST_FLY)))
write.csv(stages_plot, file.path(OUT, "stage_values_plotlevel.csv"), row.names = FALSE)

## 9.2 stage values per water regime x treatment ------------------------------
stages_trt <- group_pars %>%
  group_by(Index, Water_Regime, Treatment) %>%
  group_modify(function(d, k) stage_from_par(as.list(d[1, ]))) %>%
  ungroup()
write.csv(stages_trt, file.path(OUT, "stage_values_by_treatment.csv"), row.names = FALSE)

## 9.3 one curve per index x water regime, treatments pooled ------------------
# Treatment effects on the parameters are not significant (PART 6), so the
# regime-level curve is the honest summary for a stage figure.
starts_wr <- params %>% group_by(Index, Water_Regime) %>%
  summarise(across(c(y0, y1, t1, t2, k1, k2), mean), .groups = "drop")

wr_fit <- map_dfr(seq_len(nrow(starts_wr)), function(i) {
  g  <- starts_wr[i, ]
  df <- long %>% filter(Index == g$Index, Water_Regime == g$Water_Regime)
  f  <- fit_group(df, g[, c("y0","y1","t1","t2","k1","k2")])
  if (is.null(f)) return(NULL)
  as_tibble(as.list(coef(f))) %>%
    mutate(Index = g$Index, Water_Regime = g$Water_Regime)
})

stages_wr <- wr_fit %>% group_by(Index, Water_Regime) %>%
  group_modify(function(d, k) stage_from_par(as.list(d[1, ]))) %>% ungroup()
write.csv(left_join(wr_fit, stages_wr, by = c("Index","Water_Regime"),
                    suffix = c("", "_stage")),
          file.path(OUT, "stage_values_by_regime.csv"), row.names = FALSE)

cat("\n-- stage dates by water regime (DAS) --\n")
print(as.data.frame(stages_wr %>% mutate(across(where(is.numeric), ~ round(.x, 1)))))

## 9.4 regime contrast for every stage ---------------------------------------
stage_tests <- stages_plot %>%
  pivot_longer(all_of(stage_cols), names_to = "Stage", values_to = "Value") %>%
  group_by(Index, Stage) %>%
  group_modify(function(d, key) {
    vI <- d$Value[d$Water_Regime == "I"]; vD <- d$Value[d$Water_Regime == "D"]
    pp <- splitplot_p(d)
    tibble(I = mean(vI), D = mean(vD), diff = mean(vI) - mean(vD),
           cohen_d = (mean(vI) - mean(vD)) / sqrt((var(vI) + var(vD)) / 2),
           p_regime = pp["p_regime"], p_treatment = pp["p_treatment"],
           p_interaction = pp["p_interaction"],
           p_wilcox = wilcox.test(Value ~ Water_Regime, data = d, exact = FALSE)$p.value)
  }) %>% ungroup() %>%
  group_by(Stage) %>% mutate(p_wilcox_BH = p.adjust(p_wilcox, "BH")) %>% ungroup()
write.csv(stage_tests, file.path(OUT, "stage_tests.csv"), row.names = FALSE)

## 9.5 stage panel figure, all indices ---------------------------------------
DISPLAY_MAX <- 92                      # matches the earlier trial's x range
grid_disp   <- seq(min(spec$DAS), DISPLAY_MAX, by = 0.25)

curves_wr <- wr_fit %>%
  group_by(Index, Water_Regime) %>%
  group_modify(function(d, k) {
    p <- as.list(d[1, ])
    tibble(DAS = grid_disp,
           fit = dbl_log(grid_disp, p$y0, p$y1, p$t1, p$t2, p$k1, p$k2))
  }) %>% ungroup() %>%
  mutate(Part = ifelse(DAS <= LAST_FLY, "observed window", "extrapolated"))

obs_wr <- long %>% group_by(Index, Water_Regime, DAS) %>%
  summarise(m = mean(Value), se = sd(Value) / sqrt(n()), .groups = "drop")

marks <- stages_wr %>%
  select(Index, Water_Regime, SOS, t1, EOG, POS, SOD, t2, EOS) %>%
  pivot_longer(-c(Index, Water_Regime), names_to = "Stage", values_to = "DAS") %>%
  mutate(Stage = factor(Stage, levels = c("SOS","t1","EOG","POS","SOD","t2","EOS")))

# per-panel annotation: the drought-to-irrigated shift in the three key dates
ann <- stages_wr %>%
  select(Index, Water_Regime, t1, POS, t2) %>%
  pivot_longer(-c(Index, Water_Regime), names_to = "Stage", values_to = "v") %>%
  pivot_wider(names_from = Water_Regime, values_from = v) %>%
  mutate(d = I - D) %>%
  group_by(Index) %>%
  summarise(lab = sprintf("shift: t1 %+.1f d   POS %+.1f d   t2 %+.1f d",
                          d[Stage == "t1"], d[Stage == "POS"], d[Stage == "t2"]),
            .groups = "drop")

rng <- long %>% group_by(Index) %>%
  summarise(ymin = min(Value), ymax = max(Value), .groups = "drop") %>%
  left_join(ann, by = "Index") %>%
  mutate(y = ymax + 0.06 * (ymax - ymin), DAS = min(spec$DAS))

stage_lt <- c(SOS = "dotted", t1 = "dashed", EOG = "dotted", POS = "solid",
              SOD = "dotted", t2 = "dashed", EOS = "dotted")

p_stage <- ggplot() +
  geom_vline(data = marks, aes(xintercept = DAS, colour = Water_Regime,
                               linetype = Stage), linewidth = 0.35, alpha = 0.75) +
  geom_line(data = curves_wr, aes(DAS, fit, colour = Water_Regime, linewidth = Part)) +
  geom_errorbar(data = obs_wr, aes(DAS, ymin = m - se, ymax = m + se,
                                   colour = Water_Regime), width = 1.2, linewidth = 0.35) +
  geom_point(data = obs_wr, aes(DAS, m, colour = Water_Regime), size = 1.3) +
  geom_text(data = rng, aes(DAS, y, label = lab), hjust = 0, size = 2.5) +
  geom_blank(data = rng, aes(DAS, y * 1.01)) +   # room for the label, per panel
  facet_wrap(~ Index, scales = "free_y", ncol = 3) +
  scale_colour_manual(values = wr_col, labels = wr_lab, name = NULL) +
  scale_linetype_manual(values = stage_lt, name = "Stage") +
  scale_linewidth_manual(values = c(`observed window` = 0.85, extrapolated = 0.35),
                         name = NULL) +
  scale_x_continuous(breaks = seq(20, 90, 10)) +
  labs(x = "Days after sowing",
       y = "Index value",
       title = "Double-logistic canopy trajectories and phenological stages",
       subtitle = paste0("vertical lines: SOS, t1, EOG, POS, SOD, t2, EOS | ",
                         "thin curve beyond ", LAST_FLY,
                         " DAS is extrapolated; shifts are irrigated minus drought")) +
  guides(linetype = guide_legend(nrow = 1, override.aes = list(colour = "grey30")),
         linewidth = guide_legend(nrow = 1)) +
  theme_pheno
ggsave(file.path(FIG, "Fig_multiindex_stages.png"), p_stage, width = 11, height = 7.5, dpi = 400)
ggsave(file.path(FIG, "Fig_multiindex_stages.pdf"), p_stage, width = 11, height = 7.5)

## 9.6 one detailed stage figure per index -----------------------------------
fig_stage_one <- function(idx) {
  cw <- curves_wr %>% filter(Index == idx)
  ow <- obs_wr    %>% filter(Index == idx)
  mk <- marks     %>% filter(Index == idx)
  yr <- range(c(cw$fit, ow$m)); pad <- diff(yr)
  lab <- mk %>% mutate(y = yr[1] - 0.04 * pad -
                           0.05 * pad * (Water_Regime == "D"))
  ggplot() +
    geom_vline(data = mk, aes(xintercept = DAS, colour = Water_Regime,
                              linetype = Stage), linewidth = 0.4, alpha = 0.8) +
    geom_line(data = cw, aes(DAS, fit, colour = Water_Regime, linewidth = Part)) +
    geom_errorbar(data = ow, aes(DAS, ymin = m - se, ymax = m + se,
                                 colour = Water_Regime), width = 1.2, linewidth = 0.35) +
    geom_point(data = ow, aes(DAS, m, colour = Water_Regime), size = 1.6) +
    geom_text(data = lab, aes(DAS, y, label = sprintf("%s\n%.0f", Stage, DAS),
                              colour = Water_Regime), size = 2.2, vjust = 1) +
    scale_colour_manual(values = wr_col, labels = wr_lab, name = NULL) +
    scale_linetype_manual(values = stage_lt, name = "Stage") +
    scale_linewidth_manual(values = c(`observed window` = 0.9, extrapolated = 0.4),
                           name = NULL) +
    scale_x_continuous(breaks = seq(20, 90, 10)) +
    expand_limits(y = yr[1] - 0.16 * pad) +
    labs(x = "Days after sowing", y = idx,
         title = paste0(idx, ": fitted phenological stages"),
         subtitle = paste0("thin curve beyond ", LAST_FLY, " DAS is extrapolated")) +
    guides(linetype = guide_legend(nrow = 1, override.aes = list(colour = "grey30"))) +
    theme_pheno + theme(legend.box = "vertical", legend.margin = margin(0, 0, 0, 0))
}

for (idx in REPORT) {
  safe <- gsub("[^A-Za-z0-9]", "", idx)
  ggsave(file.path(FIG, paste0("Fig_stages_", safe, ".png")), fig_stage_one(idx),
         width = 9, height = 6, dpi = 400)
}

## 9.7 stage dates as a timeline --------------------------------------------
p_timeline <- marks %>%
  ggplot(aes(DAS, Index, colour = Water_Regime, shape = Stage)) +
  geom_vline(xintercept = LAST_FLY, linetype = 3, colour = "grey50") +
  geom_point(size = 2.2, position = position_dodge(width = 0.5)) +
  scale_colour_manual(values = wr_col, labels = wr_lab, name = NULL) +
  scale_shape_manual(values = c(SOS = 1, t1 = 16, EOG = 2, POS = 8,
                                SOD = 5, t2 = 15, EOS = 0), name = "Stage") +
  scale_x_continuous(breaks = seq(10, 110, 10)) +
  labs(x = "Days after sowing", y = NULL,
       title = "Fitted phenological stage dates",
       subtitle = "dotted line: last flight (83 DAS); marks to its right are extrapolated") +
  theme_pheno
ggsave(file.path(FIG, "Fig_stage_timeline.png"), p_timeline, width = 9, height = 5, dpi = 400)

message("done. outputs in ", normalizePath(OUT))

