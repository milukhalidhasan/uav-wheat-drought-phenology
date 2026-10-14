# ---------------------------------------------------------------------------
# Machine learning sweep: models x predictor sets x validation schemes.
# R port of 01_Code/run_ml_analysis.py / run_ml_extended.py.
#
# Nested cross-validation throughout: hyper-parameters are tuned by an inner
# k-fold INSIDE every outer fold, so no tuning decision ever sees the fold it
# is scored on.
#
# Dependencies: glmnet, pls, ranger, e1071, kernlab, FNN  (gbm optional)
#   source("install_packages.R")  installs them.
#
# Run:  Rscript 03_ml_analysis.R
#
# RUNTIME WARNING ---------------------------------------------------------
# The full Python sweep (5,148 configurations) took ~4.5 h on 12 cores.
# R is slower, so this script ships with a REDUCED default configuration
# (see the CONFIG block) that finishes in roughly 20-40 min single-threaded.
# Widen FEATURE_SETS / TARGETS_RUN / MODELS_RUN to go further.
# ---------------------------------------------------------------------------
source("00_config.R")
need(c("glmnet", "pls", "ranger", "e1071", "kernlab", "FNN"))

# ------------------------------ CONFIG -------------------------------------
# Each can be overridden from the shell, e.g.
#   WHEAT_MODELS="Ridge,Lasso"  WHEAT_SETS="Pheno"  Rscript 03_ml_analysis.R
env_or <- function(var, default)
  if (nzchar(Sys.getenv(var))) trimws(strsplit(Sys.getenv(var), ",")[[1]]) else default

TARGETS_RUN  <- env_or("WHEAT_TARGETS", c("Grain_Yield_t_ha"))
FEATURE_SETS <- env_or("WHEAT_SETS",
                       c("Design_only", "Agro_predictive", "Pheno", "Spectral"))
MODELS_RUN   <- env_or("WHEAT_MODELS",
                       c("Dummy_mean", "OLS", "Ridge", "Lasso", "ElasticNet", "PLSR",
                         "SVR_RBF", "RandomForest", "ExtraTrees", "GPR", "KNN"))
SCHEMES_RUN  <- env_or("WHEAT_SCHEMES",
                       c("CV_2024-25", "Block_2024-25", "CV_2025-26", "Block_2025-26",
                         "CV_pooled", "Transfer_24to25", "Transfer_25to24",
                         "TransferZ_24to25", "TransferZ_25to24"))
N_SPLITS  <- 5L
N_REPEATS <- as.integer(Sys.getenv("WHEAT_REPEATS", "3"))
N_INNER   <- as.integer(Sys.getenv("WHEAT_INNER", "4"))
OUT_NAME  <- Sys.getenv("WHEAT_OUT", "ml_results_R")
# ---------------------------------------------------------------------------

SUF <- if (file.exists(file.path(CSVD, "Plot_Design_R.csv"))) "_R" else ""
rdc <- function(n) read.csv(file.path(CSVD, paste0(n, SUF, ".csv")),
                            check.names = FALSE, stringsAsFactors = FALSE)

# ---------------------------------------------------------------------------
# data + predictor blocks
# ---------------------------------------------------------------------------
load_data <- function() {
  agro <- rdc("ML_Agronomic_CrossSeason")
  phen <- rdc("ML_Phenology_CrossSeason")
  spec <- rdc("Spectral_Interp_CommonDAS")
  pheno_feats <- grep("_R2$", setdiff(names(phen), c(ML_KEYS, TARGETS)),
                      value = TRUE, invert = TRUE)
  agro_all  <- setdiff(names(agro), c(ML_KEYS, TARGETS))
  agro_pred <- setdiff(agro_all, COMPONENTS)
  spec_feats <- setdiff(names(spec), c("Season","Season_ID","Plot_UID","Plot_ID",
                                       "Water_Regime","Replication","Treatment"))
  df <- merge(agro, phen[, c("Plot_UID", pheno_feats)], by = "Plot_UID")
  df <- merge(df, spec[, c("Plot_UID", spec_feats)], by = "Plot_UID")
  stopifnot(nrow(df) == 192)
  sets <- list(
    Design_only     = list(num = character(0),        cat = DESIGN_CAT),
    Agro_predictive = list(num = agro_pred,           cat = character(0)),
    Agro_full       = list(num = agro_all,            cat = character(0)),
    Pheno           = list(num = pheno_feats,         cat = character(0)),
    Spectral        = list(num = spec_feats,          cat = character(0)),
    All_parameters  = list(num = c(agro_pred, pheno_feats, spec_feats), cat = DESIGN_CAT))
  list(df = df[order(df$Season_ID, df$Plot_ID), ], sets = sets)
}

usable <- function(d, cols) cols[vapply(cols, function(c0)
  c0 %in% names(d) && sum(!is.na(d[[c0]])) > 2 &&
    length(unique(d[[c0]][!is.na(d[[c0]])])) > 1, logical(1))]

# ---------------------------------------------------------------------------
# design matrix: standardise numerics on TRAIN statistics, one-hot the factors
# ---------------------------------------------------------------------------
make_xy <- function(tr, te, num, cat, target) {
  num <- setdiff(num, FORBIDDEN[[target]])
  Xtr <- if (length(num)) as.matrix(tr[, num, drop = FALSE]) else matrix(nrow = nrow(tr), ncol = 0)
  Xte <- if (length(num)) as.matrix(te[, num, drop = FALSE]) else matrix(nrow = nrow(te), ncol = 0)
  if (ncol(Xtr)) {
    mu <- colMeans(Xtr, na.rm = TRUE)
    sg <- apply(Xtr, 2, sd, na.rm = TRUE); sg[!is.finite(sg) | sg == 0] <- 1
    Xtr <- sweep(sweep(Xtr, 2, mu, "-"), 2, sg, "/")
    Xte <- sweep(sweep(Xte, 2, mu, "-"), 2, sg, "/")
    Xtr[!is.finite(Xtr)] <- 0; Xte[!is.finite(Xte)] <- 0
  }
  for (c0 in cat) {
    lv <- sort(unique(as.character(tr[[c0]])))[-1]      # drop first level
    for (l in lv) {
      Xtr <- cbind(Xtr, as.numeric(as.character(tr[[c0]]) == l))
      Xte <- cbind(Xte, as.numeric(as.character(te[[c0]]) == l))
      colnames(Xtr)[ncol(Xtr)] <- colnames(Xte)[ncol(Xte)] <- paste0(c0, "_", l)
    }
  }
  list(Xtr = Xtr, Xte = Xte,
       ytr = as.numeric(tr[[target]]), yte = as.numeric(te[[target]]))
}

# ---------------------------------------------------------------------------
# the model zoo: each returns predictions for Xte, tuning internally on Xtr
# ---------------------------------------------------------------------------
kfold <- function(n, k, seed) {
  set.seed(seed); split(sample(n), rep(seq_len(k), length.out = n))
}

fit_predict <- function(model, Xtr, ytr, Xte, inner = N_INNER) {
  p <- ncol(Xtr)
  if (model == "Dummy_mean") return(rep(mean(ytr), nrow(Xte)))

  if (model == "OLS") {
    fit <- lm.fit(cbind(1, Xtr), ytr)
    b <- fit$coefficients; b[is.na(b)] <- 0
    return(as.numeric(cbind(1, Xte) %*% b))
  }
  if (model %in% c("Ridge", "Lasso", "ElasticNet")) {
    alpha <- c(Ridge = 0, Lasso = 1, ElasticNet = 0.5)[[model]]
    nf <- max(3, min(inner, floor(length(ytr) / 8)))
    cv <- glmnet::cv.glmnet(Xtr, ytr, alpha = alpha, nfolds = nf,
                            standardize = FALSE, family = "gaussian")
    return(as.numeric(predict(cv, newx = Xte, s = "lambda.min")))
  }
  if (model == "PLSR") {
    maxc <- min(15, p, length(ytr) - 2)
    if (maxc < 1) return(rep(mean(ytr), nrow(Xte)))
    d <- data.frame(y = ytr, X = I(Xtr))
    fit <- pls::plsr(y ~ X, data = d, ncomp = maxc, validation = "CV",
                     segments = min(inner, 5))
    nc <- pls::selectNcomp(fit, method = "onesigma", plot = FALSE)
    if (is.na(nc) || nc < 1) nc <- which.min(as.numeric(pls::RMSEP(fit)$val[1, 1, -1]))
    return(as.numeric(predict(fit, newdata = data.frame(X = I(Xte)), ncomp = nc)))
  }
  if (model == "SVR_RBF") {
    best <- NULL; bestc <- Inf
    folds <- kfold(length(ytr), min(inner, 4), SEED)
    for (C in c(1, 10, 100)) for (g in c(1 / max(p, 1), 0.01, 0.1)) {
      err <- 0
      for (f in folds) {
        m <- e1071::svm(Xtr[-f, , drop = FALSE], ytr[-f], kernel = "radial",
                        cost = C, gamma = g, epsilon = 0.1, scale = FALSE)
        err <- err + sum((ytr[f] - predict(m, Xtr[f, , drop = FALSE]))^2)
      }
      if (err < bestc) { bestc <- err; best <- c(C, g) }
    }
    m <- e1071::svm(Xtr, ytr, kernel = "radial", cost = best[1], gamma = best[2],
                    epsilon = 0.1, scale = FALSE)
    return(as.numeric(predict(m, Xte)))
  }
  if (model %in% c("RandomForest", "ExtraTrees")) {
    rule <- if (model == "ExtraTrees") "extratrees" else "variance"
    best <- NULL; bestc <- Inf
    for (mt in unique(pmax(1, round(c(0.3, 0.7) * p)))) {
      m <- ranger::ranger(x = Xtr, y = ytr, num.trees = 300, mtry = mt,
                          splitrule = rule, seed = SEED, num.threads = 1)
      if (m$prediction.error < bestc) { bestc <- m$prediction.error; best <- mt }
    }
    m <- ranger::ranger(x = Xtr, y = ytr, num.trees = 300, mtry = best,
                        splitrule = rule, seed = SEED, num.threads = 1)
    return(predict(m, Xte)$predictions)
  }
  if (model == "GBM") {
    if (!has_pkg("gbm")) return(rep(mean(ytr), nrow(Xte)))
    d <- data.frame(y = ytr, Xtr)
    m <- gbm::gbm(y ~ ., data = d, distribution = "gaussian", n.trees = 400,
                  interaction.depth = 3, shrinkage = 0.05, bag.fraction = 0.8,
                  cv.folds = 0, verbose = FALSE)
    return(predict(m, data.frame(Xte), n.trees = 400))
  }
  if (model == "GPR") {
    m <- try(kernlab::gausspr(Xtr, ytr, kernel = "rbfdot", variance.model = FALSE,
                              scaled = FALSE), silent = TRUE)
    if (inherits(m, "try-error")) return(rep(mean(ytr), nrow(Xte)))
    return(as.numeric(kernlab::predict(m, Xte)))
  }
  if (model == "KNN") {
    best <- 5; bestc <- Inf
    folds <- kfold(length(ytr), min(inner, 4), SEED)
    for (k in c(3, 5, 7, 10)) {
      err <- 0
      for (f in folds) {
        pr <- FNN::knn.reg(train = Xtr[-f, , drop = FALSE], test = Xtr[f, , drop = FALSE],
                           y = ytr[-f], k = min(k, length(ytr) - length(f)))$pred
        err <- err + sum((ytr[f] - pr)^2)
      }
      if (err < bestc) { bestc <- err; best <- k }
    }
    return(FNN::knn.reg(train = Xtr, test = Xte, y = ytr, k = best)$pred)
  }
  stop("unknown model: ", model)
}

# ---------------------------------------------------------------------------
# validation schemes
# ---------------------------------------------------------------------------
metrics <- function(y, yhat)
  c(R2 = r2_score(y, yhat), RMSE = rmse(y, yhat), MAE = mae(y, yhat),
    bias = mean(yhat - y))

eval_cv <- function(model, d, num, cat, target, groups = NULL) {
  y <- as.numeric(d[[target]]); n <- nrow(d)
  if (is.null(groups)) {
    reps <- lapply(seq_len(N_REPEATS), function(r) kfold(n, N_SPLITS, SEED + r))
  } else {
    g <- as.character(d[[groups]])
    reps <- list(lapply(unique(g), function(u) which(g == u)))
  }
  per <- t(vapply(reps, function(folds) {
    oof <- rep(NA_real_, n)
    for (f in folds) {
      xy <- make_xy(d[-f, , drop = FALSE], d[f, , drop = FALSE], num, cat, target)
      oof[f] <- fit_predict(model, xy$Xtr, xy$ytr, xy$Xte)
    }
    metrics(y, oof)
  }, numeric(4)))
  c(setNames(colMeans(per), paste0(colnames(per), "_mean")),
    setNames(apply(per, 2, sd), paste0(colnames(per), "_sd")))
}

eval_transfer <- function(model, tr, te, num, cat, target, zscore = FALSE) {
  num <- setdiff(num, FORBIDDEN[[target]])
  if (zscore) {
    zs <- function(d) {
      for (nm in c(num, target)) {
        v <- suppressWarnings(as.numeric(d[[nm]]))
        s <- sd(v, na.rm = TRUE)
        if (!is.finite(s) || s == 0) s <- 1
        d[[nm]] <- (v - mean(v, na.rm = TRUE)) / s
      }
      d
    }
    tr <- zs(tr); te <- zs(te)
  }
  xy <- make_xy(tr, te, num, cat, target)
  pred <- fit_predict(model, xy$Xtr, xy$ytr, xy$Xte)
  m <- metrics(xy$yte, pred)
  c(setNames(m, paste0(names(m), "_mean")),
    setNames(rep(0, 4), paste0(names(m), "_sd")))
}

# ---------------------------------------------------------------------------
main <- function() {
  dat <- load_data(); df <- dat$df; sets <- dat$sets
  d1 <- df[df$Season == S1, ]; d2 <- df[df$Season == S2, ]

  jobs <- expand.grid(Target = TARGETS_RUN, Feature_set = FEATURE_SETS,
                      Model = MODELS_RUN, Scheme = SCHEMES_RUN,
                      stringsAsFactors = FALSE)
  cat(nrow(jobs), "configurations\n")
  t0 <- Sys.time(); out <- vector("list", nrow(jobs))

  for (i in seq_len(nrow(jobs))) {
    j <- jobs[i, ]
    fs <- sets[[j$Feature_set]]
    n1 <- usable(d1, fs$num); n2 <- usable(d2, fs$num); nb <- intersect(n1, n2)
    r <- try(switch(j$Scheme,
      "CV_2024-25"       = eval_cv(j$Model, d1, n1, fs$cat, j$Target),
      "CV_2025-26"       = eval_cv(j$Model, d2, n2, fs$cat, j$Target),
      "Block_2024-25"    = eval_cv(j$Model, d1, n1, fs$cat, j$Target, groups = "Replication"),
      "Block_2025-26"    = eval_cv(j$Model, d2, n2, fs$cat, j$Target, groups = "Replication"),
      "CV_pooled"        = eval_cv(j$Model, df, nb, c(fs$cat, "Season"), j$Target),
      "Transfer_24to25"  = eval_transfer(j$Model, d1, d2, nb, fs$cat, j$Target),
      "Transfer_25to24"  = eval_transfer(j$Model, d2, d1, nb, fs$cat, j$Target),
      "TransferZ_24to25" = eval_transfer(j$Model, d1, d2, nb, fs$cat, j$Target, TRUE),
      "TransferZ_25to24" = eval_transfer(j$Model, d2, d1, nb, fs$cat, j$Target, TRUE)),
      silent = TRUE)
    row <- data.frame(j, n_features = length(nb) + length(fs$cat),
                      stringsAsFactors = FALSE)
    if (inherits(r, "try-error")) {
      row$error <- as.character(r)
    } else {
      for (k in names(r)) row[[k]] <- unname(r[[k]])
    }
    out[[i]] <- row
    if (i %% 10 == 0 || i == nrow(jobs))
      cat(sprintf("  %4d/%4d  %.1f min elapsed\n", i, nrow(jobs),
                  as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  }

  cols <- unique(unlist(lapply(out, names)))
  res <- do.call(rbind, lapply(out, function(r) {
    for (c0 in setdiff(cols, names(r))) r[[c0]] <- NA
    r[, cols, drop = FALSE]
  }))
  write_tbl(res, OUT_NAME)
  cat("\nwrote", file.path(RESD, paste0(OUT_NAME, ".csv")), "\n\n")

  ok <- res[!is.na(res$R2_mean) & res$Model != "Dummy_mean", ]
  if (nrow(ok)) {
    best <- do.call(rbind, lapply(split(ok, list(ok$Feature_set, ok$Scheme), drop = TRUE),
                                  function(g) g[which.max(g$R2_mean), ]))
    tab <- tapply(best$R2_mean, list(best$Feature_set, best$Scheme), identity)
    cat("BEST R2 per predictor set x scheme (", TARGETS_RUN[1], ")
", sep = "")
    print(round(tab[intersect(FEATURE_SETS, rownames(tab)),
                    intersect(SCHEMES_RUN, colnames(tab)), drop = FALSE], 3))
  } else {
    cat("no non-baseline models were run; see ", OUT_NAME, ".csv
", sep = "")
  }
  invisible(res)
}

res <- main()
