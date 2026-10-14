"""Analysis A, follow-ups.

1. Within-season standardisation (TrZ) is label-free but assumes the new season contains drought
   and irrigated plots in similar proportion. Test it when drought plots are a minority of the
   new season (50 %, 33 %, 20 %), drawing random subsets of the drought plots 30 times.
2. Direction of the senescence shift: mean t2 (mid-senescence) and green duration by regime.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from a_drought_classifier import fit, models, score
from common import DL_INDICES, RESULTS, SEASONS, label, load_dl, zscore_within

warnings.filterwarnings("ignore")


def imbalance():
    dl, cols = load_dl()
    dl = dl.dropna(subset=cols).reset_index(drop=True)
    rng = np.random.default_rng(0)
    rows = []
    for a, b in [(SEASONS[0], SEASONS[1]), (SEASONS[1], SEASONS[0])]:
        tr = zscore_within(dl[dl.Season == a].reset_index(drop=True), cols)
        fitted = {n: fit(e, g, tr[cols].to_numpy(), label(tr), tr["Replication"].to_numpy())
                  for n, (e, g) in models().items()}
        te_all = dl[dl.Season == b].reset_index(drop=True)
        irr, dro = te_all[te_all.Water_Regime == "I"], te_all[te_all.Water_Regime == "D"]
        for n_d in [48, 24, 12]:
            for rep in range(1 if n_d == 48 else 30):
                sub = pd.concat([irr, dro.sample(n_d, random_state=int(rng.integers(1e9)))]).reset_index(drop=True)
                te = zscore_within(sub, cols)
                X, y = te[cols].to_numpy(), label(te)
                for n, m in fitted.items():
                    rows.append({"direction": f"{a} -> {b}", "drought_share_pct": round(100 * n_d / (48 + n_d)),
                                 "repeat": rep, "model": n,
                                 "balanced_acc": balanced_accuracy_score(y, m.predict(X)),
                                 "auc": roc_auc_score(y, score(m, X))})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "A6_TrZ_class_imbalance_all.csv", index=False)
    summ = (res.groupby(["direction", "drought_share_pct", "model"])[["balanced_acc", "auc"]].mean().reset_index())
    summ.to_csv(RESULTS / "A6_TrZ_class_imbalance_summary.csv", index=False)
    return summ


def senescence_shift():
    dl, _ = load_dl()
    rows = []
    for s in SEASONS:
        d = dl[dl.Season == s]
        for i in DL_INDICES:
            for p in ["t2_mid_senescence", "Green_Duration", "AUC_common_19_83"]:
                c = f"{i}_{p}"
                mp = d.groupby(["Replication", "Water_Regime"])[c].mean().unstack()
                diff = mp["D"] - mp["I"]
                rows.append({"Season": s, "index": i, "parameter": p,
                             "irrigated": d.loc[d.Water_Regime == "I", c].mean(),
                             "drought": d.loc[d.Water_Regime == "D", c].mean(),
                             "D_minus_I": diff.mean(), "reps_D_lower": int((diff < 0).sum())})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "A7_senescence_shift_by_regime.csv", index=False)
    return res


if __name__ == "__main__":
    s = imbalance()
    print(s.pivot_table(index=["direction", "drought_share_pct"], columns="model", values="balanced_acc").round(3).to_string())
    print(s.groupby(["direction", "drought_share_pct"])[["balanced_acc", "auc"]].mean().round(3).to_string())
    print(senescence_shift().round(2).to_string())
