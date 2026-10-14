"""
Conceptual figures the manuscript needs that can be derived from the data or the
model itself (as opposed to the site map and weather panels, which need external
inputs and are left as placeholders in the manuscript).

  figMS1_dl_schematic.png    annotated double-logistic curve: every parameter and
                             every derived stage, on one diagram
  figMS2_workflow.png        the analysis workflow from raw imagery to validation
  figMS3_layout.png          schematic field layout of the split/factorial design
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from wheat_paths import FIGD

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 400, "font.size": 9,
                     "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.bbox": "tight"})
C_D, C_I = "#C05621", "#2B6CB0"
LN9 = np.log(9)


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def dl(t, y0, A, t1, t2, k1, k2):
    return y0 + A * (sig(k1 * (t - t1)) - sig(k2 * (t - t2)))


# ---------------------------------------------------------------------------
def fig_schematic():
    y0, A, t1, t2, k1, k2 = 0.18, 0.62, 38.0, 80.0, 0.16, 0.14
    t = np.linspace(0, 125, 800)
    y = dl(t, y0, A, t1, t2, k1, k2)
    SOS, EOG = t1 - LN9 / k1, t1 + LN9 / k1
    SOD, EOS = t2 - LN9 / k2, t2 + LN9 / k2
    POS = t[np.argmax(y)]
    peak = y.max()

    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    m = (t >= 19) & (t <= 83)
    ax.fill_between(t[m], y0, y[m], color="#68D391", alpha=0.22, lw=0,
                    label="AUC over the common window (19–83 DAS)")
    ax.plot(t, y, color="#1A365D", lw=2.6, zorder=5)
    ax.axhline(y0, color="grey", ls=":", lw=1.1)
    ax.annotate("", xy=(112, y0), xytext=(112, peak),
                arrowprops=dict(arrowstyle="<->", color="#9B2226", lw=1.5))
    ax.text(113.5, (y0 + peak) / 2, "A\namplitude", color="#9B2226",
            fontsize=9, va="center", fontweight="bold")
    ax.text(2, y0 - 0.045, "y₀  baseline", color="grey", fontsize=9, fontweight="bold")

    for x, lab, col in [(SOS, "SOS", "#2F855A"), (EOG, "EOG", "#2F855A"),
                        (POS, "POS", "#B7791F"), (SOD, "SOD", "#9B2226"),
                        (EOS, "EOS", "#9B2226")]:
        ax.plot([x, x], [y0 - 0.02, dl(x, y0, A, t1, t2, k1, k2)], color=col,
                ls="--", lw=1.0, alpha=0.8)
        ax.plot(x, dl(x, y0, A, t1, t2, k1, k2), "o", color=col, ms=6, zorder=6)
        ax.text(x, y0 - 0.055, lab, color=col, ha="center", fontsize=8.5,
                fontweight="bold")

    for x, lab in [(t1, "t₁"), (t2, "t₂")]:
        ax.plot(x, dl(x, y0, A, t1, t2, k1, k2), "s", color="#1A365D", ms=7, zorder=7)
        ax.text(x, dl(x, y0, A, t1, t2, k1, k2) + 0.035, lab, ha="center",
                fontsize=10, fontweight="bold", color="#1A365D")

    # tangents illustrating the rates
    for x, k, lab in [(t1, k1, "slope = A·k₁/4\n(green-up rate)"),
                      (t2, k2, "slope = A·k₂/4\n(senescence rate)")]:
        s = A * k / 4 * (1 if x == t1 else -1)
        yc = dl(x, y0, A, t1, t2, k1, k2)
        xs = np.array([x - 11, x + 11])
        ax.plot(xs, yc + s * (xs - x), color="#B7791F", lw=1.6, alpha=0.9)
        ax.text(x + (-32 if x == t1 else 6), yc + (0.12 if x == t1 else 0.12), lab,
                fontsize=8, color="#B7791F", fontweight="bold")

    ax.annotate("", xy=(t1, y0 - 0.095), xytext=(t2, y0 - 0.095),
                arrowprops=dict(arrowstyle="<->", color="#1A365D", lw=1.3))
    ax.text((t1 + t2) / 2, y0 - 0.125, "green duration  (t₂ − t₁)", ha="center",
            fontsize=8.5, color="#1A365D", fontweight="bold")
    ax.annotate("", xy=(SOS, y0 - 0.16), xytext=(EOS, y0 - 0.16),
                arrowprops=dict(arrowstyle="<->", color="grey", lw=1.3))
    ax.text((SOS + EOS) / 2, y0 - 0.19, "length of season  (EOS − SOS)", ha="center",
            fontsize=8.5, color="grey", fontweight="bold")

    ax.set_xlabel("Days after sowing (DAS)")
    ax.set_ylabel("Vegetation index")
    ax.set_ylim(y0 - 0.23, peak + 0.13)
    ax.set_xlim(0, 125)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.set_title("Double-logistic model and the parameters derived from it",
                 fontsize=11, fontweight="bold", loc="left")
    fig.savefig(os.path.join(FIGD, "figMS1_dl_schematic.png"))
    plt.close(fig)
    print("  figMS1_dl_schematic")


# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc, ec="#2D3748", fs=8.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.45)


def arrow(ax, p1, p2, col="#2D3748"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                                 color=col, lw=1.2, shrinkA=2, shrinkB=2))


def fig_workflow():
    fig, ax = plt.subplots(figsize=(10.6, 7.4), constrained_layout=True)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    B, G, Y, P, R = "#EBF4FF", "#E6FFFA", "#FFFAF0", "#FAF5FF", "#FFF5F5"

    box(ax, 0.2, 8.7, 4.4, 1.0,
        "FIELD EXPERIMENT\n2 water regimes × 4 N rates × 3 seaweed doses × 4 reps\n"
        "96 plots per season × 2 seasons = 192 plot-seasons", B, bold=True, fs=8.0)
    box(ax, 5.4, 8.7, 4.4, 1.0,
        "UAV MULTISPECTRAL FLIGHTS\nMicaSense Altum · 11 flights (2024-25, 10–108 DAS)\n"
        "10 flights (2025-26, 19–83 DAS)", G, bold=True, fs=8.0)

    box(ax, 0.2, 7.0, 2.05, 1.3, "Destructive\nharvest traits\nyield, components,\n"
                                 "H₂O₂, MDA", B, fs=7.8)
    box(ax, 2.45, 7.0, 2.15, 1.3, "Ground series\nchlorophyll, height,\nsoil temperature,\n"
                                  "leaf physiology", B, fs=7.8)
    box(ax, 5.4, 7.0, 2.05, 1.3, "Orthomosaics\nplot-level\nreflectance\n5 bands", G, fs=7.8)
    box(ax, 7.65, 7.0, 2.15, 1.3, "38 vegetation\nindices\nper plot\nper flight", G, fs=7.8)

    arrow(ax, (2.4, 8.7), (1.2, 8.3)); arrow(ax, (2.4, 8.7), (3.5, 8.3))
    arrow(ax, (7.6, 8.7), (6.4, 8.3)); arrow(ax, (7.6, 8.7), (8.7, 8.3))

    box(ax, 5.4, 5.3, 2.05, 1.3, "Double-logistic\nfitting\nEq. (1)\n6 indices", Y, fs=7.8)
    box(ax, 7.65, 5.3, 2.15, 1.3, "Linear interpolation\nto common dates\n20–80 DAS\n"
                                  "(7 dates)", Y, fs=7.8)
    arrow(ax, (6.4, 7.0), (6.4, 6.6)); arrow(ax, (8.7, 7.0), (8.7, 6.6))

    box(ax, 0.2, 5.3, 4.4, 1.3,
        "CROSS-SEASON HARMONISATION\n• reconcile the two parameterisations\n"
        "• recompute all stages with one formula\n"
        "• re-integrate AUC over the shared 19–83 DAS window", Y, bold=True, fs=7.9)
    arrow(ax, (1.2, 7.0), (1.2, 6.6)); arrow(ax, (3.5, 7.0), (3.5, 6.6))
    arrow(ax, (5.4, 5.95), (4.6, 5.95))

    box(ax, 0.2, 3.5, 9.6, 1.4,
        "PREDICTOR BLOCKS  (522 parameters)\n"
        "Treatment (3)  ·  Agronomic (14)  ·  Physiology (4)  ·  Ground (7–38)  ·  "
        "Double-logistic (138)  ·  Spectral (294)  ·  Hybrid (152)  ·  All (446)",
        P, bold=True, fs=8.2)
    arrow(ax, (2.4, 5.3), (2.4, 4.9)); arrow(ax, (6.4, 5.3), (6.4, 4.9))
    arrow(ax, (8.7, 5.3), (8.7, 4.9))

    box(ax, 0.2, 1.7, 4.6, 1.5,
        "12 ALGORITHMS\nDummy · OLS · Ridge · Lasso · ElasticNet\n"
        "PLSR · SVR-RBF · RF · ExtraTrees\nHistGB · GPR · kNN", R, bold=True, fs=8.0)
    box(ax, 5.2, 1.7, 4.6, 1.5,
        "5 VALIDATION SCHEMES\nCV · Blk (leave-one-rep-out) · Pooled\n"
        "Tr (between-season, raw)\nTrZ (between-season, standardised)", R, bold=True, fs=8.0)
    arrow(ax, (2.5, 3.5), (2.5, 3.2)); arrow(ax, (7.5, 3.5), (7.5, 3.2))

    box(ax, 2.2, 0.15, 5.6, 1.1,
        "NESTED CROSS-VALIDATION\ninner 4-fold tuning inside every outer fold\n"
        "6,444 model configurations", "#E2E8F0", bold=True, fs=8.6)
    arrow(ax, (2.5, 1.7), (4.0, 1.25)); arrow(ax, (7.5, 1.7), (6.0, 1.25))

    ax.set_title("Analytical workflow", fontsize=12, fontweight="bold", loc="left")
    fig.savefig(os.path.join(FIGD, "figMS2_workflow.png"))
    plt.close(fig)
    print("  figMS2_workflow")


# ---------------------------------------------------------------------------
def fig_layout():
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0), constrained_layout=True)
    for ax, (wr, col, name) in zip(axes, [("D", C_D, "Managed drought (D)"),
                                          ("I", C_I, "Irrigated (I)")]):
        for rep in range(4):
            order = rng.permutation(12)              # treatments randomised within block
            for pos, trt in enumerate(order):
                x, y = pos, 3 - rep
                shade = 0.25 + 0.6 * (trt // 3) / 3.0
                ax.add_patch(Rectangle((x, y), 0.92, 0.88,
                                       fc=col, alpha=shade, ec="white", lw=1.4))
                ax.text(x + 0.46, y + 0.44, "T%d" % trt, ha="center", va="center",
                        fontsize=7.4, color="white", fontweight="bold")
            ax.text(-0.35, y + 0.44, "R%d" % (rep + 1), ha="right", va="center",
                    fontsize=9, fontweight="bold")
        ax.set_xlim(-1.1, 12.2); ax.set_ylim(-0.35, 4.35)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(name + "   ·   48 plots", fontsize=10.5, fontweight="bold")
    from matplotlib.patches import Patch
    handles = [Patch(fc="grey", alpha=0.25 + 0.6 * i / 3.0,
                     label="N%d  (%d kg ha⁻¹)" % (i, [0, 210, 300, 390][i]))
               for i in range(4)]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                   ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle("Schematic field layout: 12 treatments (T0–T11) randomised within each "
                 "of four replications, in each water regime",
                 fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(FIGD, "figMS3_layout.png"))
    plt.close(fig)
    print("  figMS3_layout")


if __name__ == "__main__":
    fig_schematic()
    fig_workflow()
    fig_layout()
    print("figures ->", FIGD)
