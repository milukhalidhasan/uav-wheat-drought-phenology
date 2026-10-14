# ---------------------------------------------------------------------------
# Run the whole R pipeline in order.
#   Rscript run_all.R
# Set WHEAT_SKIP_ML=1 to skip the (slow) machine learning sweep.
# ---------------------------------------------------------------------------
a <- commandArgs(trailingOnly = FALSE)
m <- grep("^--file=", a, value = TRUE)
if (length(m)) setwd(dirname(normalizePath(sub("^--file=", "", m[1]))))
steps <- c("01_build_combined_dataset.R", "02_univariate_analysis.R")
if (!nzchar(Sys.getenv("WHEAT_SKIP_ML"))) steps <- c(steps, "03_ml_analysis.R")
steps <- c(steps, "04_figures.R")
for (s in steps) {
  cat("\n==================================================\n", s,
      "\n==================================================\n", sep = "")
  t0 <- Sys.time()
  source(s, local = new.env())
  cat(sprintf("[%s done in %.1f min]\n", s,
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))
}
cat("\nPipeline complete.\n")
