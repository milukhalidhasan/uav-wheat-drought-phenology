"""
Single source of truth for every path in the pipeline.

By default the project root is inferred from where this file lives:

    <PROJECT_ROOT>/
        Wheat_Analysis_2024/05_Dataset/Wheat_2024-25_Structured_Dataset.xlsx
        Wheat_Analysis_2025/MegaData2025_Cleaned.xlsx
        Combined_TwoSeason/
            01_Code/wheat_paths.py      <- this file
            02_Dataset/  03_CSV/  04_ML/

Override either location without editing code:

    set WHEAT_ROOT=D:\\my\\project          (Windows)
    export WHEAT_ROOT=/home/me/project      (Linux / macOS)

and optionally WHEAT_S1_XLSX / WHEAT_S2_XLSX to point at the two source workbooks
directly if they sit somewhere else.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
COMBINED = os.path.dirname(_HERE)                       # .../Combined_TwoSeason
ROOT = os.environ.get("WHEAT_ROOT", os.path.dirname(COMBINED))

F24 = os.environ.get("WHEAT_S1_XLSX", os.path.join(
    ROOT, "Wheat_Analysis_2024", "05_Dataset", "Wheat_2024-25_Structured_Dataset.xlsx"))
F25 = os.environ.get("WHEAT_S2_XLSX", os.path.join(
    ROOT, "Wheat_Analysis_2025", "MegaData2025_Cleaned.xlsx"))

OUT = COMBINED
DATASET_DIR = os.path.join(COMBINED, "02_Dataset")
CSVD = os.path.join(COMBINED, "03_CSV")
ML_DIR = os.path.join(COMBINED, "04_ML")
RESD = os.path.join(ML_DIR, "results")
FIGD = os.path.join(ML_DIR, "figures")
XLSX = os.path.join(DATASET_DIR, "Wheat_TwoSeason_Combined.xlsx")
REPORT_DOCX = os.path.join(COMBINED, "Wheat_TwoSeason_Analysis_Report.docx")

for _d in (DATASET_DIR, CSVD, ML_DIR, RESD, FIGD):
    os.makedirs(_d, exist_ok=True)


def check_inputs():
    missing = [p for p in (F24, F25) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Source workbook(s) not found:\n  " + "\n  ".join(missing) +
            "\n\nSet WHEAT_ROOT (or WHEAT_S1_XLSX / WHEAT_S2_XLSX) to the right location.")
    return True


if __name__ == "__main__":
    print("ROOT        :", ROOT)
    print("season 1 xlsx:", F24, "  exists:", os.path.exists(F24))
    print("season 2 xlsx:", F25, "  exists:", os.path.exists(F25))
    for n, p in [("dataset", DATASET_DIR), ("csv", CSVD), ("results", RESD),
                 ("figures", FIGD)]:
        print("%-12s:" % n, p)
