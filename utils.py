"""
utils.py - shared helpers: wide-table columns, naming, BH correction, saving.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

import config as C


def time_cols(df, prefix="Time_"):
    tc = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    return sorted(tc, key=lambda s: int(s.split("_")[-1]))


def meta_cols(df):
    return [c for c in df.columns if not (isinstance(c, str) and c.startswith("Time_"))]


def ensure(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_csv(df, name, outdir):
    if df is None or df.empty:
        print(f"  [skip] {name}: empty")
        return
    ensure(outdir)
    df.to_csv(os.path.join(outdir, f"{name}.csv"), index=False)
    print(f"  OK {name}.csv  ({len(df):,} rows x {len(df.columns)} cols)")


def savefig(fig, stem, outdir, dpi=300):
    ensure(outdir)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"), dpi=dpi,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {stem}.png/.svg")


def base_of(col):
    return col.rsplit("_", 1)[0] if col.endswith(tuple(f"_{s}" for s in C.STATS)) else col


def category(col):
    return C.CATEGORY.get(base_of(col), "state")


def pretty(col):
    return C.PRETTY.get(base_of(col), base_of(col))


def star(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def add_bh(t, family=None, pcol="p_raw"):
    """Benjamini-Hochberg within each family; adds p_BH, sig_raw, sig_BH."""
    t = t.copy()
    t["p_BH"] = np.nan
    if t.empty:
        return t
    for _, idx in t.groupby(family or C.FAMILY).groups.items():
        p = t.loc[idx, pcol].dropna()
        if len(p):
            t.loc[p.index, "p_BH"] = multipletests(p.values, method="fdr_bh")[1]
    t["sig_raw"] = t[pcol].apply(star)
    t["sig_BH"] = t["p_BH"].apply(star)
    return t


def split_condition(df):
    df = df.copy()
    df[["noise_type", "treatment"]] = df["Condition"].str.rsplit("_", n=1, expand=True)
    return df


def load_summaries():
    """Per-animal summary (all stages/setups) and adult house-split summary."""
    main = pd.read_csv(os.path.join(C.FEAT_DIR, "per_animal_summary.csv"), low_memory=False)
    house = pd.read_csv(os.path.join(C.FEAT_DIR, "adult_house_split_summary.csv"))
    for name, d in (("main", main), ("house", house)):
        n_dup = d.duplicated(C.KEYS).sum()
        if n_dup:
            raise ValueError(f"{name} summary has {n_dup} duplicated animal rows")
    return split_condition(main), split_condition(house)


def group_data(group, main, house):
    return house if group.startswith("adult_") and "house" in group else main


def retained(group):
    """PCA-retained metrics of a group, from 03_feature_selection_pca output."""
    L = pd.read_csv(os.path.join(C.PCA_DIR, "pca_loadings_by_stage.csv"))
    return L.loc[L["Stage"] == group, "Metric"].tolist()
