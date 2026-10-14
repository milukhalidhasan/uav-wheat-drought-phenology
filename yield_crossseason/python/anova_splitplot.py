"""
Split-plot ANOVA with the correct error strata.

The trial is a split plot: water regime sits on the main plots and the twelve
nitrogen x seaweed combinations on the subplots, within four replications. The
earlier analysis in this project tested every term against the residual, which
is the wrong error term for the main-plot factor and makes the water-regime test
anti-conservative.

Correct structure (per season, n = 96):

    Source                  df   tested against
    Replication              3   -
    Water regime             1   Replication x Water regime   (main-plot error)
    Rep x Water (error a)    3
    Treatment               11   residual (error b)
    Water x Treatment       11   residual
    Residual (error b)      66

Nitrogen and seaweed are also tested separately as a 4 x 3 factorial within the
subplot stratum, which is what the agronomic question actually asks.

Writes results/treatment_effects_splitplot.csv and a comparison against the
earlier residual-error test.
"""
import os
import itertools

import numpy as np
import pandas as pd
from scipy import stats

import extended_features as E
from wheat_paths import RESD

S1, S2 = E.S1, E.S2


def bh(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full(p.shape, np.nan)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = np.minimum.accumulate((pv[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def dummies(series):
    lv = sorted(series.unique())[1:]
    return np.column_stack([(series == l).astype(float) for l in lv]) if lv \
        else np.zeros((len(series), 0))


def inter(A, B):
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.zeros((A.shape[0], 0))
    return np.column_stack([A[:, i] * B[:, j]
                            for i in range(A.shape[1]) for j in range(B.shape[1])])


def rss_rank(X, y):
    A = np.column_stack([np.ones(len(y))] + ([X] if X.shape[1] else []))
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.sum((y - A @ beta) ** 2)), np.linalg.matrix_rank(A)


def splitplot(d, col):
    """Return dict of F and p for Water, N, Seaweed, NxS, WxN, WxS, Treatment,
    WxTreatment using the proper split-plot error strata."""
    y = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
    keep = np.isfinite(y)
    if keep.sum() < 60 or np.nanstd(y[keep]) == 0:
        return None
    d = d[keep]
    y = y[keep]

    R = dummies(d.Replication)
    W = dummies(d.Water_Regime)
    N = dummies(d.N_Level)
    S = dummies(d.Seaweed_Level)
    T = dummies(d.Treatment)
    RW = inter(R, W)                      # main-plot error stratum

    def ss(terms):
        X = np.column_stack([t for t in terms if t.shape[1]]) if terms else \
            np.zeros((len(y), 0))
        return rss_rank(X, y)

    # ---- main-plot stratum ----
    # Water is tested after Replication; the main-plot error is the Rep x Water
    # interaction fitted AFTER both main effects. Fitting Rep x Water before
    # Water would absorb most of the water effect into the error term.
    r_r,   k_r   = ss([R])                 # Replication only
    r_rw_, k_rw_ = ss([R, W])              # + Water
    r_sat, k_sat = ss([R, W, RW])          # saturated main-plot model
    df_w = k_rw_ - k_r
    df_a = k_sat - k_rw_                   # main-plot error df (= 3 here)
    ms_a = (r_rw_ - r_sat) / df_a if df_a > 0 else np.nan
    out = {}
    if df_w > 0 and df_a > 0 and ms_a > 0:
        ms_w = (r_r - r_rw_) / df_w
        F = ms_w / ms_a
        out["F_Water"] = F
        out["p_Water"] = float(stats.f.sf(F, df_w, df_a))
        out["df_Water"] = "%d, %d" % (df_w, df_a)

    # ---- subplot stratum: everything else tested against residual ----
    NS, WN, WS = inter(N, S), inter(W, N), inter(W, S)
    named = [("R", R), ("W", W), ("RW", RW), ("N", N), ("Seaweed", S),
             ("NxS", NS), ("WxN", WN), ("WxS", WS)]
    full_terms = [b for _, b in named]
    r_full, k_full = ss(full_terms)
    dfe = len(y) - k_full
    if dfe <= 0 or r_full <= 0:
        return out
    mse = r_full / dfe
    for name in ["N", "Seaweed", "NxS", "WxN", "WxS"]:
        red = [b for nm, b in named if nm != name]
        r_red, k_red = ss(red)
        q = k_full - k_red
        if q <= 0:
            continue
        F = ((r_red - r_full) / q) / mse
        out["F_" + name] = F
        out["p_" + name] = float(stats.f.sf(F, q, dfe))
        out["df_" + name] = "%d, %d" % (q, dfe)

    # ---- treatment as a single 12-level subplot factor ----
    WT = inter(W, T)
    tnamed = [("R", R), ("W", W), ("RW", RW), ("Treatment", T), ("WxTreatment", WT)]
    tt = [b for _, b in tnamed]
    r_t, k_t = ss(tt)
    dfe_t = len(y) - k_t
    if dfe_t > 0 and r_t > 0:
        mse_t = r_t / dfe_t
        for name in ["Treatment", "WxTreatment"]:
            red = [b for nm, b in tnamed if nm != name]
            r_red, k_red = ss(red)
            q = k_t - k_red
            if q <= 0:
                continue
            F = ((r_red - r_t) / q) / mse_t
            out["F_" + name] = F
            out["p_" + name] = float(stats.f.sf(F, q, dfe_t))
    out["R2_model"] = 1 - r_full / float(np.sum((y - y.mean()) ** 2))
    return out


def main():
    df, blocks = E.build()
    params = [c for c in df.columns
              if df[c].dtype.kind in "fi"
              and c not in ("Season_ID", "N_kg_ha", "Seaweed_pct")]
    params = [c for c in params
              if df[c].notna().sum() > 20 and df[c].nunique(dropna=True) > 1]
    print("testing %d parameters with split-plot error strata" % len(params))

    rows = []
    for season in (S1, S2):
        d0 = df[df.Season == season]
        for c in params:
            r = splitplot(d0, c)
            if r:
                r.update({"Season": season, "Parameter": c})
                rows.append(r)
    out = pd.DataFrame(rows)
    terms = ["Water", "N", "Seaweed", "NxS", "WxN", "WxS", "Treatment", "WxTreatment"]
    for t in terms:
        pc = "p_" + t
        if pc in out:
            out["q_" + t] = np.nan
            for s in (S1, S2):
                m = (out.Season == s) & out[pc].notna()
                if m.any():
                    out.loc[m, "q_" + t] = bh(out.loc[m, pc])
    lead = ["Season", "Parameter", "R2_model"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    f = os.path.join(RESD, "treatment_effects_splitplot.csv")
    out.to_csv(f, index=False)
    print("wrote", f)

    print("\nSIGNIFICANT PARAMETERS (q < 0.05) UNDER THE CORRECT SPLIT-PLOT ERROR")
    tab = pd.DataFrame({s: {t: int((out.loc[out.Season == s, "q_" + t] < 0.05).sum())
                            for t in terms if "q_" + t in out}
                        for s in (S1, S2)})
    print(tab.to_string())

    old = os.path.join(RESD, "treatment_effects.csv")
    if os.path.exists(old):
        o = pd.read_csv(old)
        print("\nCOMPARISON with the earlier residual-error (Type II) test")
        cmp = []
        for t in ["Water", "N", "Seaweed"]:
            for s in (S1, S2):
                a = int((o.loc[o.Season == s, "q_" + t] < 0.05).sum()) \
                    if "q_" + t in o else np.nan
                b = int((out.loc[out.Season == s, "q_" + t] < 0.05).sum()) \
                    if "q_" + t in out else np.nan
                cmp.append({"Term": t, "Season": s,
                            "residual error (wrong for Water)": a,
                            "split-plot error (correct)": b})
        print(pd.DataFrame(cmp).to_string(index=False))

    key = ["Grain_Yield_t_ha", "Biological_Yield_t_ha", "Harvest_Index",
           "Chlorophyll_mean", "H2O2_umol_per_gFW", "NDVI_DAS80",
           "NDVI_AUC_common_19_83"]
    print("\nKEY TRAITS, split-plot p and q")
    k = out[out.Parameter.isin(key)][
        ["Season", "Parameter", "p_Water", "q_Water", "p_N", "q_N",
         "p_Seaweed", "q_Seaweed"]]
    print(k.round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    main()
