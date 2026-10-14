"""
Univariate screening of EVERY available parameter against every yield target, plus
the factorial treatment analysis. This is the descriptive layer that sits under the
ML: which single variables move with yield, in which season, and does the treatment
structure move them.

Outputs
  results/univariate_correlations.csv   Pearson + Spearman vs each target, per season,
                                        with BH-adjusted p and cross-season consistency
  results/univariate_top.csv            the strongest, consistent associations
  results/treatment_effects.csv         3-way ANOVA (water x N x seaweed) per parameter
  results/season_comparison.csv         season means, SD, Welch t and Hedges g
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

import extended_features as E

from wheat_paths import (ROOT, F24, F25, OUT, CSVD, XLSX, RESD, FIGD,
                         ML_DIR, REPORT_DOCX, check_inputs)

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
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def block_of(col, blocks):
    for name, cols in blocks.items():
        if col in cols:
            return name
    return "other"


def correlations(df, blocks, params):
    rows = []
    for target in E.TARGETS:
        for season in (S1, S2):
            d = df[df.Season == season]
            y = d[target].to_numpy(float)
            for c in params:
                if c == target or c in E.FORBIDDEN[target]:
                    continue
                x = d[c].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 10 or np.nanstd(x[ok]) == 0:
                    continue
                r, pr = stats.pearsonr(x[ok], y[ok])
                rho, prho = stats.spearmanr(x[ok], y[ok])
                rows.append({"Target": target, "Season": season, "Parameter": c,
                             "Block": block_of(c, blocks), "n": int(ok.sum()),
                             "pearson_r": r, "pearson_p": pr,
                             "spearman_rho": rho, "spearman_p": prho})
    out = pd.DataFrame(rows)
    for (t, s), g in out.groupby(["Target", "Season"]):
        m = (out.Target == t) & (out.Season == s)
        out.loc[m, "pearson_q_BH"] = bh(out.loc[m, "pearson_p"])
        out.loc[m, "spearman_q_BH"] = bh(out.loc[m, "spearman_p"])

    # cross-season consistency: same sign and both significant after BH
    piv = out.pivot_table(index=["Target", "Parameter", "Block"], columns="Season",
                          values=["pearson_r", "pearson_q_BH"])
    cons = pd.DataFrame(index=piv.index)
    cons["r_2024_25"] = piv[("pearson_r", S1)]
    cons["r_2025_26"] = piv[("pearson_r", S2)]
    cons["q_2024_25"] = piv[("pearson_q_BH", S1)]
    cons["q_2025_26"] = piv[("pearson_q_BH", S2)]
    cons["same_sign"] = np.sign(cons.r_2024_25) == np.sign(cons.r_2025_26)
    cons["both_sig_BH05"] = (cons.q_2024_25 < 0.05) & (cons.q_2025_26 < 0.05)
    cons["mean_abs_r"] = (cons.r_2024_25.abs() + cons.r_2025_26.abs()) / 2
    cons["consistent"] = cons.same_sign & cons.both_sig_BH05
    return out, cons.reset_index()


TERM_FACTORS = {"Water": {"W"}, "N": {"N"}, "Seaweed": {"S"},
                "WxN": {"W", "N"}, "WxS": {"W", "S"}, "NxS": {"N", "S"},
                "WxNxS": {"W", "N", "S"}}


def anova3(df, params):
    """Three-way ANOVA per parameter, per season: water regime x nitrogen x seaweed.

    Type II sums of squares: a term is tested against the model containing every
    term that does NOT include it. Testing a main effect while its own interactions
    stay in the model would make that test almost meaningless, so the reduced model
    drops the term together with every higher-order term containing it.
    """
    import itertools
    rows = []
    for season in (S1, S2):
        d0 = df[df.Season == season]
        for c in params:
            y_all = d0[c].to_numpy(float)
            keep = np.isfinite(y_all)
            if keep.sum() < 40 or np.nanstd(y_all[keep]) == 0:
                continue
            d = d0[keep]
            y = y_all[keep]
            X = pd.get_dummies(d[["Water_Regime", "N_Level", "Seaweed_Level"]],
                               drop_first=True).astype(float)
            main = {"W": X[[k for k in X.columns if k.startswith("Water_Regime")]],
                    "N": X[[k for k in X.columns if k.startswith("N_Level")]],
                    "S": X[[k for k in X.columns if k.startswith("Seaweed_Level")]]}
            blocks = {"Water": main["W"], "N": main["N"], "Seaweed": main["S"]}
            for nm, (a, b) in [("WxN", ("W", "N")), ("WxS", ("W", "S")),
                               ("NxS", ("N", "S"))]:
                cols = {"%s:%s" % (ca, cb): main[a][ca] * main[b][cb]
                        for ca, cb in itertools.product(main[a].columns, main[b].columns)}
                blocks[nm] = pd.DataFrame(cols, index=d.index)
            cols3 = {}
            for ca, cb, cc in itertools.product(main["W"].columns, main["N"].columns,
                                                main["S"].columns):
                cols3["%s:%s:%s" % (ca, cb, cc)] = main["W"][ca] * main["N"][cb] * main["S"][cc]
            blocks["WxNxS"] = pd.DataFrame(cols3, index=d.index)

            order = ["Water", "N", "Seaweed", "WxN", "WxS", "NxS", "WxNxS"]
            full = pd.concat([blocks[t] for t in order], axis=1)

            def rss(design):
                parts = [np.ones((len(d), 1))]
                if design.shape[1]:
                    parts.append(design.to_numpy(float))
                A = np.hstack(parts)
                beta, *_ = np.linalg.lstsq(A, y, rcond=None)
                return float(((y - A @ beta) ** 2).sum()), np.linalg.matrix_rank(A)

            rss_full, k_full = rss(full)
            dfe = len(d) - k_full
            tss = float(((y - y.mean()) ** 2).sum())
            rec = {"Season": season, "Parameter": c, "n": int(len(d)), "df_resid": dfe,
                   "R2_model": 1 - rss_full / tss if tss > 0 else np.nan}
            if dfe <= 0 or rss_full <= 0:
                rows.append(rec)
                continue
            mse = rss_full / dfe
            for t in order:
                fac = TERM_FACTORS[t]
                # Type II: reduced model keeps only terms that do not contain this term
                keep_terms = [o for o in order if not fac.issubset(TERM_FACTORS[o])]
                red = pd.concat([blocks[o] for o in keep_terms], axis=1) if keep_terms \
                    else pd.DataFrame(index=d.index)
                add = pd.concat([red, blocks[t]], axis=1)
                rss_red, k_red = rss(red)
                rss_add, k_add = rss(add)
                q = k_add - k_red
                if q <= 0:
                    continue
                F = ((rss_red - rss_add) / q) / mse
                rec["F_" + t] = F
                rec["p_" + t] = float(stats.f.sf(F, q, dfe))
            rows.append(rec)
    out = pd.DataFrame(rows)
    for term in ["Water", "N", "Seaweed", "WxN", "WxS", "NxS", "WxNxS"]:
        col = "p_" + term
        if col in out:
            for s, g in out.groupby("Season"):
                out.loc[out.Season == s, "q_" + term] = bh(out.loc[out.Season == s, col])
    return out


def season_compare(df, params):
    rows = []
    for c in params:
        a = df.loc[df.Season == S1, c].to_numpy(float)
        b = df.loc[df.Season == S2, c].to_numpy(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 5 or len(b) < 5:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
        g = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
        g *= 1 - 3 / (4 * (len(a) + len(b)) - 9)
        rows.append({"Parameter": c, "mean_2024_25": a.mean(), "sd_2024_25": a.std(ddof=1),
                     "mean_2025_26": b.mean(), "sd_2025_26": b.std(ddof=1),
                     "welch_t": t, "p": p, "hedges_g": g})
    out = pd.DataFrame(rows)
    out["q_BH"] = bh(out.p)
    return out


def main():
    os.makedirs(RESD, exist_ok=True)
    df, blocks = E.build()
    params = [c for c in df.columns
              if df[c].dtype.kind in "fi" and c not in ("Season_ID", "N_kg_ha", "Seaweed_pct")]
    params = [c for c in params if df[c].notna().sum() > 20 and df[c].nunique(dropna=True) > 1]
    print("screening %d parameters" % len(params))

    corr, cons = correlations(df, blocks, params)
    corr.to_csv(os.path.join(RESD, "univariate_correlations.csv"), index=False)
    cons.to_csv(os.path.join(RESD, "univariate_consistency.csv"), index=False)

    top = (cons[cons.Target == "Grain_Yield_t_ha"].sort_values("mean_abs_r", ascending=False))
    top.to_csv(os.path.join(RESD, "univariate_top.csv"), index=False)

    av = anova3(df, params)
    av.to_csv(os.path.join(RESD, "treatment_effects.csv"), index=False)

    sc = season_compare(df, params)
    sc.to_csv(os.path.join(RESD, "season_comparison.csv"), index=False)

    print("\nTOP 20 parameters vs GRAIN YIELD (consistent across both seasons)")
    t = top[top.consistent].head(20)
    print(t[["Parameter", "Block", "r_2024_25", "r_2025_26", "mean_abs_r"]]
          .round(3).to_string(index=False))
    print("\nconsistent associations by block (grain yield):")
    print(top[top.consistent].Block.value_counts().to_string())
    print("\nsignificant after BH in at least one season, by block:")
    sig = cons[(cons.Target == "Grain_Yield_t_ha") &
               ((cons.q_2024_25 < 0.05) | (cons.q_2025_26 < 0.05))]
    print(sig.Block.value_counts().to_string())

    print("\nTREATMENT EFFECTS: parameters with q<0.05, count by term and season")
    terms = [c for c in av.columns if c.startswith("q_")]
    print(av.groupby("Season")[terms].apply(lambda g: (g < 0.05).sum()).to_string())


if __name__ == "__main__":
    main()
