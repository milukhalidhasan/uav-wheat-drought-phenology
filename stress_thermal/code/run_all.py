"""Run every analysis and rebuild every figure (about 6 minutes on 12 cores)

Reads data/full if present, otherwise the synthetic data/demo tables.."""
import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

for script in ["a_drought_classifier.py", "a2_robustness.py", "b_thermal_analysis.py", "figures.py"]:
    print(f"\n=== {script}")
    runpy.run_path(str(HERE / script), run_name="__main__")
