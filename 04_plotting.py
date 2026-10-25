"""
04_plotting.py - figure panels.

Fig 1C/G/J, 2B  % change from own control, PCA-retained metrics (DUR, mean)
Fig 1 D-K       example trajectories during the noise window (DUR)
Fig 3B-E        noise vs own control box plots (AF, p90; raw Mann-Whitney p shown)
Fig S1          per-stage PCA scree plots and loading wheels (PC1/PC2, PC3/PC4)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import category, group_data, load_summaries, pretty, retained, savefig, star, time_cols

plt.rcParams.update(C.RC)
OUT = {k: os.path.join(C.FIG_DIR, k) for k in ("bubbles", "boxplots", "pca", "trajectories")}


# =============================================================================
# BUBBLES (% change from own control)
# =============================================================================
def pct_change_table(data, group, cols):
    sub = data[(data["Stage"] == group) & (data["Setup"] == "DUR")]
    rows = []
    for nt in C.NOISE_TYPES:
        for col in cols:
            c = pd.to_numeric(sub.loc[sub["Condition"] == f"{nt}_control", col], errors="coerce").dropna()
            n = pd.to_numeric(sub.loc[sub["Condition"] == f"{nt}_noise", col], errors="coerce").dropna()
            if len(c) < C.MIN_N or len(n) < C.MIN_N or abs(c.mean()) < 1e-9:
                continue
            rows.append(dict(Metric=col, noise_type=nt, pct_change=(n.mean() - c.mean()) / abs(c.mean()) * 100))
    return pd.DataFrame(rows)


def plot_bubble(t, cols, title, stem):
    cats = ["kinematic", "arena", "skeleton", "state"]
    order = [c for cat in cats for c in sorted(m for m in cols if category(m) == cat)]
    y = {m: len(order) - 1 - i for i, m in enumerate(order)}
    size = lambda v: min(abs(v), 150) * 9 + 10
    fig, ax = plt.subplots(figsize=(max(94, 2 * 26 + 62) * C.MM, max(70, len(order) * 6.5 + 32) * C.MM))
    for cat in cats:
        ys = [y[m] for m in order if category(m) == cat]
        if ys:
            ax.axhspan(min(ys) - 0.48, max(ys) + 0.48, color=C.CAT_BAND[cat], alpha=0.7, zorder=0, lw=0)
            ax.annotate(cat.title(), xy=(0, np.mean(ys)), xycoords=("axes fraction", "data"),
                        xytext=(-92, 0), textcoords="offset points", ha="center", va="center",
                        rotation=90, fontsize=6, fontweight="bold", color=C.CAT_COLOR[cat])
            if min(ys) > 0:
                ax.axhline(min(ys) - 0.5, color="0.55", lw=0.6, ls="--", zorder=1)
    for _, r in t.iterrows():
        ax.scatter(C.NOISE_TYPES.index(r["noise_type"]), y[r["Metric"]], s=size(r["pct_change"]),
                   c=C.INCREASE_COLOR if r["pct_change"] >= 0 else C.DECREASE_COLOR,
                   alpha=0.82, edgecolors="white", linewidths=0.35, zorder=3)
    ax.set_yticks([y[m] for m in order])
    ax.set_yticklabels([pretty(m) for m in order], fontsize=5.5)
    ax.set_xticks(range(len(C.NOISE_TYPES)))
    ax.set_xticklabels([C.NOISE_LABEL[n] for n in C.NOISE_TYPES])
    ax.set_xlim(-0.6, len(C.NOISE_TYPES) - 0.4)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.spines["left"].set_visible(False)
    ax.set_title(title, fontweight="bold")
    fig.legend(handles=[mpatches.Patch(color=C.INCREASE_COLOR, label="Increase vs control"),
                        mpatches.Patch(color=C.DECREASE_COLOR, label="Decrease vs control")],
               loc="center left", bbox_to_anchor=(0.91, 0.3))
    ax.legend(handles=[plt.scatter([], [], s=size(v), c="0.55", edgecolors="white", label=f"{v}%")
                       for v in (10, 50, 100)], title="|% change|", loc="upper left",
              bbox_to_anchor=(1.02, 1.0), labelspacing=1.6, borderpad=1.0, handletextpad=1.8)
    savefig(fig, stem, OUT["bubbles"])


def bubbles(main, house):
    for stem, g in C.FIG_BUBBLES.items():
        cols = retained(g)
        t = pct_change_table(group_data(g, main, house), g, cols)
        if t.empty:
            print(f"  [skip] {stem}: no data")
            continue
        plot_bubble(t, cols, f"{C.GROUP_LABEL[g]} (% change from control)", stem)


# =============================================================================
# BOX PLOTS (Fig 3)
# =============================================================================
def box_5_95(ax, vals, pos, color, rng, width=0.42):
    vals = vals[np.isfinite(vals)]
    p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
    h = width / 2
    ax.add_patch(plt.Rectangle((pos - h, p25), width, p75 - p25, facecolor=color, alpha=0.6,
                               edgecolor=color, lw=1.2, zorder=3))
    ax.plot([pos - h, pos + h], [p50, p50], color="white", lw=2, zorder=4, solid_capstyle="butt")
    for a, b in ((p5, p25), (p75, p95)):
        ax.plot([pos, pos], [a, b], color=color, lw=1, zorder=3)
    for cap in (p5, p95):
        ax.plot([pos - h / 2, pos + h / 2], [cap, cap], color=color, lw=1, zorder=3)
    ax.scatter(pos + rng.uniform(-h * 0.55, h * 0.55, len(vals)), np.clip(vals, p5, p95),
               color=color, alpha=0.48, s=8, lw=0, zorder=5)


def fig3(main):
    rng = np.random.default_rng(C.SEED)
    for stem, (stage, nt, bases) in C.FIG3.items():
        sub = main[(main["Stage"] == stage) & (main["Setup"] == "AF")]
        fig, axes = plt.subplots(1, len(bases), figsize=(1.6 * len(bases) + 0.4, 2.3), squeeze=False)
        for ax, b in zip(axes[0], bases):
            col = f"{b}_{C.FIG3_STAT}"
            g = {tr: pd.to_numeric(sub.loc[sub["Condition"] == f"{nt}_{tr}", col], errors="coerce")
                 .dropna().to_numpy() for tr in ("control", "noise")}
            if min(map(len, g.values())) < C.MIN_N:
                continue
            box_5_95(ax, g["control"], 1, C.CONTROL_COLOR, rng)
            box_5_95(ax, g["noise"], 2, C.NOISE_COLOR[nt], rng)
            p = mannwhitneyu(g["noise"], g["control"], alternative="two-sided").pvalue
            top = max(np.percentile(v, 95) for v in g.values())
            span = top - min(np.percentile(v, 5) for v in g.values())
            ax.plot([1, 2], [top + 0.06 * span] * 2, color="0.2", lw=0.8)
            ax.text(1.5, top + 0.08 * span, f"{star(p)} " + (f"p={p:.3f}" if p >= 0.001 else "p<0.001"), ha="center", va="bottom", fontsize=6)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Control", C.NOISE_LABEL[nt]])
            ax.set_xlim(0.4, 2.6)
            ax.set_ylabel(pretty(col))
            ax.grid(axis="y", lw=0.3, alpha=0.4)
        fig.suptitle(f"{C.NOISE_LABEL[nt]} {stage} (AF, per-animal {C.FIG3_STAT})", fontsize=8)
        fig.tight_layout()
        savefig(fig, stem, OUT["boxplots"])


# =============================================================================
# FIG S1 (PCA)
# =============================================================================
def supp_pca():
    L = pd.read_csv(os.path.join(C.PCA_DIR, "pca_loadings_by_stage.csv"))
    V = pd.read_csv(os.path.join(C.PCA_DIR, "pca_variance_by_stage.csv"))
    for g in C.SUPP_PCA_STAGES:
        v, l = V[V["Stage"] == g], L[L["Stage"] == g]
        if v.empty:
            continue
        fig, ax = plt.subplots(figsize=(110 * C.MM, 80 * C.MM))
        pcs = np.arange(1, len(v) + 1)
        ax.bar(pcs, v["Variance"] * 100, color="0.45", edgecolor="white", lw=0.4)
        tw = ax.twinx()
        tw.plot(pcs, v["Cumulative"] * 100, "o-", color="#CC3311", lw=1, ms=3)
        tw.set_ylim(0, 105)
        tw.set_ylabel("Cumulative (%)", color="#CC3311")
        tw.spines["right"].set_visible(True)
        ax.set(xlabel="PC", ylabel="Variance (%)", xticks=pcs, title=C.GROUP_LABEL[g])
        savefig(fig, f"FigS1_scree_{g}", OUT["pca"])
        for a, b in ((1, 2), (3, 4)):
            if f"PC{b}_loading" in l.columns:
                loading_wheel(l, v, g, a, b)


def loading_wheel(l, v, g, a, b):
    ev = v.set_index("PC")["Variance"] * 100
    fig, ax = plt.subplots(figsize=(110 * C.MM, 110 * C.MM))
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color="0.82", lw=0.8, ls="--")
    for _, r in l.iterrows():
        x, y = r[f"PC{a}_loading"], r[f"PC{b}_loading"]
        col, strong = C.CAT_COLOR[r["Category"]], np.hypot(x, y) >= 0.30
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4 if strong else 0.6,
                                    alpha=1 if strong else 0.35))
        if strong:
            ax.text(x * 1.15, y * 1.15, pretty(r["Metric"]), fontsize=5, ha="center", va="center",
                    color=col, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=0.5))
    ax.axhline(0, color="0.75", lw=0.5, ls="--")
    ax.axvline(0, color="0.75", lw=0.5, ls="--")
    ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal",
           xlabel=f"PC{a} ({ev[f'PC{a}']:.1f}%)", ylabel=f"PC{b} ({ev[f'PC{b}']:.1f}%)",
           title=f"{C.GROUP_LABEL[g]} PCA loadings PC{a}/PC{b}")
    ax.legend(handles=[mpatches.Patch(color=c, label=k.title()) for k, c in C.CAT_COLOR.items()],
              loc="lower left")
    savefig(fig, f"FigS1_loadings_{g}_PC{a}{b}", OUT["pca"])


# =============================================================================
# TRAJECTORIES (Fig 1)
# =============================================================================
N_BINS, JUMP_UM, MIN_STEP_UM = 40, 250.0, 0.5


def trajectories():
    cd = os.path.join(C.LOAD_DIR, "coordinates")
    wide = [pd.read_csv(os.path.join(cd, f"com_{a}_native_wide.csv"), low_memory=False) for a in "xy"]
    tc = time_cols(wide[0])
    keep = (wide[0]["Setup"] == "DUR") & (wide[0]["Part"] == wide[0]["Stage"].map(C.NOISE_PART))
    X, Y = (w.loc[keep, tc].to_numpy(float) * C.PIXEL_TO_UM for w in wide)
    meta = wide[0].loc[keep, [c for c in wide[0].columns if c not in tc]].reset_index(drop=True)
    meta["n_valid"] = np.isfinite(X).sum(1)
    cmap = mpl.colormaps["viridis"]
    for (stage, cond), g in meta.groupby(["Stage", "Condition"]):
        for i in g.nlargest(C.N_TRAJ, "n_valid").index:
            x, y = X[i].copy(), Y[i].copy()
            t = np.arange(len(x)) / C.FPS_NATIVE
            ok = np.isfinite(x)
            if ok.sum() < 2:
                continue
            x, y, t = x[ok], y[ok], t[ok]
            jump = np.r_[False, np.hypot(np.diff(x), np.diff(y)) > JUMP_UM]
            x[jump] = y[jump] = np.nan
            x, y = x - np.nanmean(x), y - np.nanmean(y)
            keep_i = [0]                                        # light SVG: decimate
            for j in range(1, len(x)):
                if not np.isfinite(x[j]) or not np.isfinite(x[keep_i[-1]]) or \
                        np.hypot(x[j] - x[keep_i[-1]], y[j] - y[keep_i[-1]]) >= MIN_STEP_UM:
                    keep_i.append(j)
            x, y, t = x[keep_i], y[keep_i], t[keep_i]
            bins = np.minimum(((t - t[0]) / max(np.ptp(t), 1e-9) * N_BINS).astype(int), N_BINS - 1)
            fig, ax = plt.subplots(figsize=(4, 4.6))
            ax.set_aspect("equal")
            ax.axis("off")
            starts = np.r_[0, np.flatnonzero(np.diff(bins)) + 1, len(bins)]
            for s, e in zip(starts[:-1], starts[1:]):          # one path per colour run
                ax.plot(x[max(s - 1, 0):e], y[max(s - 1, 0):e], lw=0.8, alpha=0.85,
                        color=cmap(bins[s] / (N_BINS - 1)), solid_capstyle="round")
            f = np.flatnonzero(np.isfinite(x))
            ax.scatter(x[f[0]], y[f[0]], marker="o", s=35, fc="white", ec="black", lw=0.8, zorder=5)
            ax.scatter(x[f[-1]], y[f[-1]], marker="X", s=40, fc="black", ec="white", lw=0.5, zorder=5)
            xl, yl = ax.get_xlim(), ax.get_ylim()
            x0, y0 = xl[0] + 0.05 * np.ptp(xl), yl[0] + 0.04 * np.ptp(yl)
            ax.plot([x0, x0 + 300], [y0, y0], color="black", lw=1.5)
            ax.text(x0 + 150, y0 + 0.02 * np.ptp(yl), "300 um", ha="center", va="bottom", fontsize=7)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, np.ptp(t) / 60))
            cb = fig.colorbar(sm, cax=fig.add_axes([0.15, 0.06, 0.7, 0.025]), orientation="horizontal")
            cb.set_label("time (min)", fontsize=6)
            r = meta.loc[i]
            ax.set_title(f"{stage} {cond}\n{r['unique_animal']} part {r['Part']}", fontsize=7, loc="left")
            savefig(fig, f"{r['unique_animal']}_part{r['Part']}", os.path.join(OUT["trajectories"], stage, cond))


if __name__ == "__main__":
    main, house = load_summaries()
    print("=== Fig 1/2 bubbles ===")
    bubbles(main, house)
    print("\n=== Fig 3 box plots ===")
    fig3(main)
    print("\n=== Fig S1 PCA ===")
    supp_pca()
    print("\n=== Fig 1 trajectories ===")
    trajectories()
    print("Done - 04_plotting")
