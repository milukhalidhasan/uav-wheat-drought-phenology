"""Run the whole Python pipeline in order.

    python run_all.py              full pipeline (the ML sweeps take hours)
    python run_all.py --skip-ml    everything except the two ML sweeps
"""
import subprocess, sys, time, os

HERE = os.path.dirname(os.path.abspath(__file__))
FAST = ["build_combined_dataset.py", "verify_yield_discrepancy.py",
        "univariate_analysis.py", "figures_phenology.py", "figures_descriptive.py"]
SLOW = ["run_ml_analysis.py", "run_ml_extended.py"]
AFTER = ["report_ml_analysis.py", "model_tables.py", "figures_extended.py"]


def run(script):
    print("\n" + "=" * 70 + "\n" + script + "\n" + "=" * 70, flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, script)], cwd=HERE)
    print("[%s -> exit %d, %.1f min]" % (script, r.returncode, (time.time() - t0) / 60),
          flush=True)
    return r.returncode


if __name__ == "__main__":
    skip = "--skip-ml" in sys.argv
    scripts = FAST + ([] if skip else SLOW) + AFTER
    if skip:
        scripts = [s for s in scripts if s not in ("report_ml_analysis.py",
                                                   "model_tables.py",
                                                   "figures_extended.py")]
    bad = [s for s in scripts if run(s) != 0]
    print("\nfailed:" if bad else "\nall steps completed.", *bad)
