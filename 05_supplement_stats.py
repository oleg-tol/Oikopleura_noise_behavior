"""
05_supplement_stats.py - statistics tables per figure and the summary supplement.

1. stats_per_figure/  Fig 1/2 bubble panels (DUR, mean), Fig 3 panels (AF, p90), Fig S1 PCA
2. centroid_shift/    noise vs own-control centroid distance in z-scored retained-feature,
                      Kaiser-PC, PC1-2 and PC3-4 space; permutation p, bootstrap range, BH
3. cliffs_delta/      Cliff's delta overview bubbles (mean and p90) + long tables
BH: within stage x setup x noise type (effect tables), within space x noise type (centroids).
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Ellipse
from scipy.stats import chi2, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import (add_bh, base_of, category, ensure, group_data, load_summaries, pretty,
                   retained, save_csv, savefig, star)

plt.rcParams.update(C.RC)
TAB_DIR = os.path.join(C.SUPP_DIR, "stats_per_figure")
CEN_DIR = os.path.join(C.SUPP_DIR, "centroid_shift")
CD_DIR  = os.path.join(C.SUPP_DIR, "cliffs_delta")
DELTA_STATS = ["mean", "p90"]
INCLUDE_AF_IN_DELTA = True
RIM = {"BH": ("black", 1.4), "raw": ("#E69F00", 1.4), "ns": ("0.70", 0.4)}
RIM_LABEL = {"BH": "BH p < 0.05", "raw": "raw p < 0.05 only", "ns": "n.s."}


# =============================================================================
# PAIRWISE EFFECT STATISTICS
# =============================================================================
_cache = {}


def compare(a, b, rng):
    """noise (a) vs control (b): descriptives, % change, Cohen's d, Cliff's delta (+boot CI), MW."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < C.MIN_N or len(b) < C.MIN_N:
        return None
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    S = np.sign(a[:, None] - b[None, :])
    ia = rng.integers(0, len(a), (C.N_BOOT, len(a)))
    ib = rng.integers(0, len(b), (C.N_BOOT, len(b)))
    boot = np.array([S[np.ix_(i, j)].mean() for i, j in zip(ia, ib)])
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    qa, qb = np.percentile(a, [25, 50, 75]), np.percentile(b, [25, 50, 75])
    return dict(n_ctrl=len(b), n_noise=len(a),
                mean_ctrl=b.mean(), sd_ctrl=b.std(ddof=1), median_ctrl=qb[1], q25_ctrl=qb[0], q75_ctrl=qb[2],
                mean_noise=a.mean(), sd_noise=a.std(ddof=1), median_noise=qa[1], q25_noise=qa[0], q75_noise=qa[2],
                pct_change=(a.mean() - b.mean()) / abs(b.mean()) * 100 if abs(b.mean()) > 1e-12 else np.nan,
                cohens_d=(a.mean() - b.mean()) / sp if sp > 0 else np.nan,
                cliffs_delta=S.mean(), delta_ci_lo=np.percentile(boot, 2.5),
                delta_ci_hi=np.percentile(boot, 97.5), U_noise=U, p_raw=p)


def effect_table(data, stage, setup, cols, rng):
    sub = data[(data["Stage"] == stage) & (data["Setup"] == setup)]
    rows = []
    for nt in C.NOISE_TYPES:
        for col in [c for c in cols if c in sub.columns]:
            key = (stage, setup, nt, col)
            if key not in _cache:
                get = lambda tr: pd.to_numeric(sub.loc[sub["Condition"] == f"{nt}_{tr}", col],
                                               errors="coerce").to_numpy(float)
                _cache[key] = compare(get("noise"), get("control"), rng)
            if _cache[key] is not None:
                rows.append({"Stage": stage, "Setup": setup, "noise_type": nt, "Noise": C.NOISE_LABEL[nt],
                             "Metric": col, "MetricPretty": pretty(col), "Category": category(col),
                             **_cache[key]})
    return pd.DataFrame(rows)


# =============================================================================
# 1. TABLES PER FIGURE
# =============================================================================
def figure_tables(main, house, rng):
    for fig, g in C.FIG_BUBBLES.items():
        t = add_bh(effect_table(group_data(g, main, house), g, "DUR", retained(g), rng))
        t.insert(0, "Figure", fig)
        t["BH_family"] = "stage x noise type, PCA-retained metrics"
        save_csv(t, fig, TAB_DIR)

    stat_cols = [c for c in main.columns if c.endswith(f"_{C.FIG3_STAT}")]
    for fig, (stage, nt, bases) in C.FIG3.items():
        t = add_bh(effect_table(main, stage, "AF", stat_cols, rng))
        t = t[t["noise_type"] == nt].copy()
        t["in_figure"] = t["Metric"].isin([f"{b}_{C.FIG3_STAT}" for b in bases])
        t["BH_family"] = f"stage x noise type, all {C.FIG3_STAT} metrics tested"
        t.insert(0, "Figure", fig)
        save_csv(t.sort_values(["in_figure", "p_raw"], ascending=[False, True]), fig, TAB_DIR)

    for f in ("pca_variance_by_stage", "pca_loadings_by_stage", "pca_feature_filter_audit"):
        shutil.copy(os.path.join(C.PCA_DIR, f"{f}.csv"),
                    os.path.join(TAB_DIR, f"FigS1_{f.replace('pca_', '').replace('_by_stage', '')}.csv"))


# =============================================================================
# 2. CENTROID SHIFT
# =============================================================================
def shift_test(A, B, rng):
    obs = np.linalg.norm(B.mean(0) - A.mean(0))
    X, nA = np.vstack([A, B]), len(A)
    null = np.empty(C.N_PERM)
    for i in range(C.N_PERM):
        idx = rng.permutation(len(X))
        null[i] = np.linalg.norm(X[idx[nA:]].mean(0) - X[idx[:nA]].mean(0))
    boot = np.array([np.linalg.norm(B[rng.integers(0, len(B), len(B))].mean(0)
                                    - A[rng.integers(0, len(A), len(A))].mean(0)) for _ in range(C.N_BOOT)])
    return dict(distance=obs, boot_lo=np.percentile(boot, 2.5), boot_hi=np.percentile(boot, 97.5),
                null_mean=null.mean(), null_q95=np.percentile(null, 95),
                z_vs_null=(obs - null.mean()) / null.std(ddof=1),
                p_perm=(1 + (null >= obs).sum()) / (C.N_PERM + 1))


def centroid_shift(main, house, rng):
    rows, store = [], {}
    L_all = pd.read_csv(os.path.join(C.PCA_DIR, "pca_loadings_by_stage.csv"))
    for g in C.PCA_GROUPS:
        cols = retained(g)
        data = group_data(g, main, house)
        sub = data[(data["Stage"] == g) & (data["Setup"] == "DUR")].reset_index(drop=True)
        if len(cols) < 2 or sub.empty:
            continue
        X = sub[cols].apply(pd.to_numeric, errors="coerce")
        Z = StandardScaler().fit_transform(X.fillna(X.median()).values)    # label-agnostic imputation
        pca = PCA(random_state=C.SEED).fit(Z)
        S = pca.transform(Z)
        Lg = L_all[L_all["Stage"] == g].set_index("Metric").reindex(cols)
        for i in range(min(4, S.shape[1])):                                # orient like Fig S1
            lc = f"PC{i + 1}_loading"
            if lc in Lg.columns and np.dot(pca.components_[i], Lg[lc].fillna(0).values) < 0:
                S[:, i] *= -1
        k = max(2, int((pca.explained_variance_ > 1).sum()))
        store[g] = (sub, S, pca)
        for nt in C.NOISE_TYPES:
            mc = (sub["Condition"] == f"{nt}_control").to_numpy()
            mn = (sub["Condition"] == f"{nt}_noise").to_numpy()
            if mc.sum() < C.MIN_N or mn.sum() < C.MIN_N:
                continue
            for space, M in (("all_features", Z), ("kaiser_PCs", S[:, :k]),
                             ("PC1-PC2", S[:, :2]), ("PC3-PC4", S[:, 2:4])):
                rows.append(dict(Group=g, GroupLabel=C.GROUP_LABEL[g], noise_type=nt,
                                 Noise=C.NOISE_LABEL[nt], space=space, n_dims=M.shape[1],
                                 n_ctrl=int(mc.sum()), n_noise=int(mn.sum()), **shift_test(M[mc], M[mn], rng)))
        print(f"  {g}: {len(cols)} features, Kaiser k={k}, n={len(sub)}")
    cs = add_bh(pd.DataFrame(rows), family=["space", "noise_type"], pcol="p_perm")
    cs["BH_family"] = "space x noise type"
    save_csv(cs, "centroid_shift_permutation", CEN_DIR)
    ct = pd.concat([plot_centroids(store, rng, (0, 1)), plot_centroids(store, rng, (2, 3))],
                   ignore_index=True)
    save_csv(ct, "centroid_coordinates_CI95", CEN_DIR)
    for sp in ("all_features", "PC1-PC2", "PC3-PC4"):
        plot_distance(cs, sp)
    return cs, ct


def plot_centroids(store, rng, pcs):
    """Centroids with 95% bootstrap ellipses; each panel on its own optimised scale."""
    sc = np.sqrt(chi2.ppf(0.95, 2))
    rows = []
    fig, axes = plt.subplots(1, len(store), figsize=(2.3 * len(store), 2.8), squeeze=False)
    for ax, (g, (sub, S, pca)) in zip(axes[0], store.items()):
        ev, ext = pca.explained_variance_ratio_ * 100, 0
        cent = {}
        for nt in C.NOISE_TYPES:
            for tr in ("control", "noise"):
                X = S[(sub["Condition"] == f"{nt}_{tr}").to_numpy()][:, list(pcs)]
                boot = np.array([X[rng.integers(0, len(X), len(X))].mean(0) for _ in range(C.N_BOOT)])
                vals, vecs = np.linalg.eigh(np.cov(boot.T))
                mu, col = X.mean(0), C.CONTROL_COLOR if tr == "control" else C.NOISE_COLOR[nt]
                ax.add_patch(Ellipse(mu, 2 * sc * np.sqrt(vals[-1]), 2 * sc * np.sqrt(vals[0]),
                                     angle=np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1])),
                                     fc=col, ec=col, alpha=0.18, lw=0.8, zorder=2))
                ax.scatter(*mu, marker=C.NOISE_MARKER[nt], s=30, color=col, ec="k", lw=0.4, zorder=4)
                cent[tr] = mu
                ext = max(ext, np.abs(mu).max() + sc * np.sqrt(vals.max()))
                lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
                rows.append(dict(Group=g, GroupLabel=C.GROUP_LABEL[g], Noise=C.NOISE_LABEL[nt], treatment=tr,
                                 n=len(X), PC_x=f"PC{pcs[0] + 1}", PC_y=f"PC{pcs[1] + 1}",
                                 var_x_pct=ev[pcs[0]], var_y_pct=ev[pcs[1]],
                                 x=mu[0], x_ci_lo=lo[0], x_ci_hi=hi[0], y=mu[1], y_ci_lo=lo[1], y_ci_hi=hi[1]))
            ax.annotate("", xy=cent["noise"], xytext=cent["control"], zorder=5,
                        arrowprops=dict(arrowstyle="->", color=C.NOISE_COLOR[nt], lw=1.3))
        lim = np.ceil(ext * 1.15 * 4) / 4
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal",
               xlabel=f"PC{pcs[0] + 1} ({ev[pcs[0]]:.1f}%)", ylabel=f"PC{pcs[1] + 1} ({ev[pcs[1]]:.1f}%)")
        ax.axhline(0, color="0.85", lw=0.5, ls="--", zorder=0)
        ax.axvline(0, color="0.85", lw=0.5, ls="--", zorder=0)
        ax.set_title(f"{C.GROUP_LABEL[g]} (n={len(sub)})", fontweight="bold")
    handles = [plt.Line2D([], [], marker=C.NOISE_MARKER[nt], ls="", color=c, mec="k", mew=0.4,
                          label=f"{C.NOISE_LABEL[nt]} {tr}")
               for nt in C.NOISE_TYPES for tr, c in (("control", C.CONTROL_COLOR), ("noise", C.NOISE_COLOR[nt]))]
    handles.append(mpatches.Patch(color="0.6", alpha=0.3, label="95% bootstrap CI of centroid"))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    savefig(fig, f"centroid_shift_PC{pcs[0] + 1}_PC{pcs[1] + 1}_CI95", CEN_DIR)
    return pd.DataFrame(rows)


def plot_distance(cs, space):
    t = cs[cs["space"] == space]
    groups = [g for g in C.PCA_GROUPS if g in set(t["Group"])]
    fig, ax = plt.subplots(figsize=(0.8 * len(groups) + 1.6, 2.8))
    for gi, g in enumerate(groups):
        for k, nt in enumerate(C.NOISE_TYPES):
            r = t[(t["Group"] == g) & (t["noise_type"] == nt)]
            if r.empty:
                continue
            r, x = r.iloc[0], gi + (k - 0.5) * 0.32
            ax.bar(x, r["null_q95"], width=0.24, color="0.86", zorder=1)
            ax.errorbar(x, r["distance"], yerr=[[max(r["distance"] - r["boot_lo"], 0)],
                                                [max(r["boot_hi"] - r["distance"], 0)]],
                        fmt=C.NOISE_MARKER[nt], color=C.NOISE_COLOR[nt], mec="k", mew=0.4,
                        ms=5, capsize=2, elinewidth=0.8, zorder=3)
            ax.text(x, max(r["boot_hi"], r["distance"]) * 1.04, star(r["p_BH"]),
                    ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([C.GROUP_LABEL[g] for g in groups], rotation=30, ha="right")
    ax.set_ylabel("Centroid distance, noise vs own control\n"
                  + {"all_features": "(z-scored retained features)",
                     "kaiser_PCs": "(Kaiser PCs)"}.get(space, f"({space} space)"))
    ax.legend(handles=[mpatches.Patch(color="0.86", label="95th pct permutation null")] +
              [plt.Line2D([], [], marker=C.NOISE_MARKER[nt], ls="", color=C.NOISE_COLOR[nt], mec="k",
                          mew=0.4, label=f"{C.NOISE_LABEL[nt]} (bootstrap 95% range)")
               for nt in C.NOISE_TYPES], loc="upper left")
    fig.tight_layout()
    savefig(fig, f"centroid_distance_vs_null_{space}", CEN_DIR)


# =============================================================================
# 3. CLIFF'S DELTA OVERVIEW
# =============================================================================
def cliffs_overview(main, house, rng, stat):
    bases = [b for b in C.METRIC_ORDER if any(base_of(m) == b for g in C.PCA_GROUPS for m in retained(g))]
    spec = [(f"{C.GROUP_LABEL[g]} | {C.NOISE_LABEL[nt]}", group_data(g, main, house), g, "DUR", nt, g)
            for g in C.PCA_GROUPS for nt in C.NOISE_TYPES]
    n_dur = len(spec)
    if INCLUDE_AF_IN_DELTA:
        spec += [(f"AF {s.title()} | {C.NOISE_LABEL[nt]}", main, s, "AF", nt, None)
                 for s in C.STAGES for nt in C.NOISE_TYPES]
    long = []
    for label, data, stage, setup, nt, g in spec:
        t = effect_table(data, stage, setup, [f"{b}_{stat}" for b in bases], rng)
        if t.empty:
            continue
        t = add_bh(t[t["noise_type"] == nt])
        t["Column"], t["Stat"], t["Base"] = label, stat, t["Metric"].map(base_of)
        t["PCA_retained"] = t["Base"].isin([base_of(m) for m in retained(g)]) if g else np.nan
        long.append(t)
    L = pd.concat(long, ignore_index=True)
    L["BH_family"] = "per column, all metrics in the overview"
    save_csv(L, f"cliffs_delta_overview_long_{stat}", CD_DIR)
    plot_cliffs_bubble(L, bases, [s[0] for s in spec], n_dur, stat)
    return L


def plot_cliffs_bubble(L, bases, labels, n_dur, stat):
    cmap = mpl.colormaps["RdBu_r"]
    norm = TwoSlopeNorm(vmin=-C.DELTA_VMAX, vcenter=0, vmax=C.DELTA_VMAX)
    y_of = {b: len(bases) - 1 - i for i, b in enumerate(bases)}
    x_of = {lab: j for j, lab in enumerate(labels)}
    size = lambda d: 10 + min(abs(d), C.DELTA_VMAX) * 380
    fig, ax = plt.subplots(figsize=(0.36 * len(labels) + 3.2, 0.28 * len(bases) + 2.6))
    for cat, band in C.CAT_BAND.items():
        ys = [y_of[b] for b in bases if category(b) == cat]
        if ys:
            ax.axhspan(min(ys) - 0.5, max(ys) + 0.5, color=band, zorder=0, lw=0)
    t = L[L["Base"].isin(y_of) & L["cliffs_delta"].notna()].copy()
    t["level"] = np.where(t["p_BH"] < 0.05, "BH", np.where(t["p_raw"] < 0.05, "raw", "ns"))
    for z, lvl in enumerate(("ns", "raw", "BH")):
        s = t[t["level"] == lvl]
        ax.scatter(s["Column"].map(x_of), s["Base"].map(y_of), s=s["cliffs_delta"].map(size),
                   c=[cmap(norm(np.clip(d, -C.DELTA_VMAX, C.DELTA_VMAX))) for d in s["cliffs_delta"]],
                   edgecolors=RIM[lvl][0], linewidths=RIM[lvl][1], zorder=3 + z)
    for j in range(2, len(labels), 2):
        ax.axvline(j - 0.5, color="0.85", lw=0.6, zorder=1)
    if n_dur < len(labels):
        ax.axvline(n_dur - 0.5, color="k", lw=1.0, zorder=1)
    ax.set(xlim=(-0.6, len(labels) - 0.4), ylim=(-0.6, len(bases) - 0.4))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks([y_of[b] for b in bases])
    ax.set_yticklabels([C.PRETTY[b] for b in bases])
    for lbl, b in zip(ax.get_yticklabels(), bases):
        lbl.set_color(C.CAT_COLOR[category(b)])
    ax.set_title(f"Cliff's delta, noise vs own control (per-animal {stat})")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=30)
    cb.set_label(f"Cliff's delta (clipped at +/-{C.DELTA_VMAX})")
    handles = [plt.scatter([], [], s=size(v), c="0.6", edgecolors="0.4", lw=0.4, label=f"|delta| = {v}")
               for v in (0.1, 0.3, 0.5)]
    handles += [plt.scatter([], [], s=size(0.3), c="white", edgecolors=RIM[k][0], lw=RIM[k][1],
                            label=RIM_LABEL[k]) for k in ("BH", "raw", "ns")]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06),
               handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    savefig(fig, f"cliffs_delta_bubble_{stat}", CD_DIR)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    ensure(TAB_DIR, CEN_DIR, CD_DIR)
    rng = np.random.default_rng(C.SEED)
    main, house = load_summaries()

    print("=== 1. Tables per figure ===")
    figure_tables(main, house, rng)

    print("\n=== 2. Centroid shift ===")
    cs, ct = centroid_shift(main, house, rng)
    print(cs.loc[cs["space"] == "all_features",
                 ["GroupLabel", "Noise", "distance", "null_q95", "p_perm", "p_BH"]].round(4).to_string(index=False))

    print("\n=== 3. Cliff's delta overview ===")
    cd = {s: cliffs_overview(main, house, rng, s) for s in DELTA_STATS}

    print("\n=== 4. Summary-supplement tables ===")
    save_csv(cs, "SuppSummary_centroid_shift_permutation", TAB_DIR)
    save_csv(ct, "SuppSummary_centroid_coordinates_CI95", TAB_DIR)
    for s, L in cd.items():
        save_csv(L, f"SuppSummary_cliffs_delta_{s}", TAB_DIR)
    print("Done - 05_supplement_stats")
