# ---------------------------------------------------------------------------
# One-time package installation.
#   Rscript install_packages.R
#
# Only 01 (readxl) and 03 (the ML packages) need anything at all.
# 02 (statistics) and 04 (figures) run on a bare R installation.
# ---------------------------------------------------------------------------
options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 600)

lib <- Sys.getenv("R_LIBS_USER")
if (!nzchar(lib) || !dir.exists(lib))
  lib <- file.path(path.expand("~"), "R", paste0(R.version$platform, "-library"),
                   paste(R.version$major, strsplit(R.version$minor, ".", fixed = TRUE)[[1]][1],
                         sep = "."))
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
cat("installing into:", lib, "\n\n")

groups <- list(
  "01 build dataset"  = c("readxl"),
  "01 xlsx output"    = c("writexl"),     # optional
  "03 machine learning" = c("glmnet", "pls", "ranger", "e1071", "kernlab", "FNN"),
  "03 optional boosting" = c("gbm"))      # optional

for (g in names(groups)) {
  pk <- groups[[g]]
  miss <- pk[!vapply(pk, requireNamespace, logical(1), quietly = TRUE)]
  cat(sprintf("%-22s %s\n", g,
              if (length(miss)) paste("installing:", paste(miss, collapse = ", "))
              else "already present"))
  if (length(miss)) try(install.packages(miss, lib = lib), silent = TRUE)
}

cat("\nfinal status\n")
for (p in unlist(groups, use.names = FALSE))
  cat(sprintf("  %-10s %s\n", p, requireNamespace(p, quietly = TRUE)))

cat("\nIf anything says FALSE, install it manually:\n",
    "  install.packages(\"<name>\")\n",
    "R version in use: ", R.version.string, "\n", sep = "")
