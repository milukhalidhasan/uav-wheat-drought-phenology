"""
The hybrid comparison the two earlier sweeps did not isolate.

Question: does combining the agronomic/physiological block with the
double-logistic block predict yield better than either block alone?

`All_parameters` in the extended sweep answers a different question, because it
also carries the 294 spectral variables, which dominate. This script runs the
clean two-block hybrid, plus a spectral-augmented variant, under exactly the same
nested-CV protocol so the numbers drop straight into the existing tables.

Feature sets added
  Agro+Pheno           agronomic/physiological  +  double-logistic
  Agro+Pheno+Design    the same, plus the treatment factors
  Agro+Spectral        agronomic/physiological  +  spectral trajectories
  Pheno+Spectral       double-logistic          +  spectral trajectories
"""
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import extended_features as E
import run_ml_extended as X
from wheat_paths import RESD

OUT = os.path.join(RESD, "ml_results_hybrid.csv")


def hybrid_sets(blocks):
    agro = blocks["Agro_predictive"] + blocks["Physio"] + blocks["Ground_summary"]
    return {
        "Agro+Pheno":        (agro + blocks["Pheno"], [], "both"),
        "Agro+Pheno+Design": (agro + blocks["Pheno"], E.DESIGN_CAT, "both"),
        "Agro+Spectral":     (agro + blocks["Spectral"], [], "both"),
        "Pheno+Spectral":    (blocks["Pheno"] + blocks["Spectral"], [], "both"),
    }


def main():
    df, blocks = E.build()
    sets = hybrid_sets(blocks)
    for name, (f, c, _) in sets.items():
        print("  %-20s %4d numeric + %d categorical" % (name, len(f), len(c)))

    jobs = []
    for target in E.TARGETS:
        for name, (feats, cat, _) in sets.items():
            zoo = X.model_zoo(len(feats) + len(cat))
            for mname, (est, grid) in zoo.items():
                for scheme in X.SCHEMES_BOTH:
                    jobs.append((df, name, feats, cat, target, mname, est, grid, scheme))
    print("\n%d jobs" % len(jobs), flush=True)

    rows = Parallel(n_jobs=-1, verbose=5, batch_size=4)(
        delayed(X.one_job)(*j) for j in jobs)
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    if "error" in res:
        bad = res[res.error.notna()]
        print("failed:", len(bad))
    print("wrote", OUT, len(res))


if __name__ == "__main__":
    main()
