"""
03_feature_selection_pca.py - feature filtering and per-stage PCA (DUR, per-animal mean).

Per group (larva, juvenile, adult, adult outside / inside house), never pooled:
NaN QC -> CV filter -> category-aware correlation filter (within, then across
categories with per-category quotas) -> z-score -> PCA (6 PCs).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import category, ensure, group_data, load_summaries, save_csv


def impute(sub, cols, how=C.IMPUTE_PCA):
    X = sub[cols].apply(pd.to_numeric, errors="coerce")
    if how == "per_condition":
        X = X.groupby(sub["Condition"]).transform(lambda s: s.fillna(s.median()))
    return X.fillna(X.median())


def select_features(sub, cols, group):
    """Returns (kept columns, audit rows)."""
    audit = []
    log = lambda m, step, val, dec: audit.append(dict(Stage=group, Metric=m, Step=step,
                                                      Value=val, Decision=dec))
    nan_frac = sub[cols].isna().mean()
    kept = [c for c in cols if nan_frac[c] <= C.NAN_FRAC_MAX]
    for c in set(cols) - set(kept):
        log(c, "nan_qc", round(nan_frac[c], 3), f"dropped NaN frac > {C.NAN_FRAC_MAX}")
    if len(kept) < 3:
        return kept, audit

    X = impute(sub, kept)
    cv = (X.std(ddof=1) / X.mean().abs().where(X.mean().abs() > 1e-9)).fillna(0)
    for c in kept:
        log(c, "cv", round(cv[c], 4), "kept" if cv[c] >= C.CV_MIN else f"dropped CV < {C.CV_MIN}")
    kept = [c for c in kept if cv[c] >= C.CV_MIN]
    if len(kept) < 3:
        return kept, audit

    r = X[kept].corr(method="spearman").abs()
    var = X[kept].var(ddof=1).sort_values(ascending=False)
    reps = {cat: [] for cat in C.CAT_COLOR}
    for cat in reps:                                    # within category, by variance
        for c in [m for m in var.index if category(m) == cat]:
            mx = r.loc[c, reps[cat]].max() if reps[cat] else np.nan
            if not reps[cat] or mx < C.CORR_MAX:
                reps[cat].append(c)
                log(c, "corr_within", round(mx, 3), "kept")
            else:
                log(c, "corr_within", round(mx, 3), f"dropped |r| with {r.loc[c, reps[cat]].idxmax()}")

    seeded = [c for cat, v in reps.items() for c in v[:C.MIN_PER_CATEGORY.get(cat, 1)]]
    rest = sorted([c for cat, v in reps.items() for c in v[C.MIN_PER_CATEGORY.get(cat, 1):]],
                  key=lambda m: var[m], reverse=True)
    final = []
    for c in seeded + rest:                             # across categories, with quotas
        mx = r.loc[c, final].max() if final else np.nan
        quota = sum(category(m) == category(c) for m in final) < C.MIN_PER_CATEGORY.get(category(c), 1)
        if not final or mx < C.CORR_MAX_CROSS or quota:
            final.append(c)
            log(c, "corr_cross", round(mx, 3), "kept (quota)" if final and mx >= C.CORR_MAX_CROSS else "kept")
        else:
            log(c, "corr_cross", round(mx, 3), f"dropped |r| with {r.loc[c, final].idxmax()}")
    return final, audit


def pinned_features():
    if not C.RETAINED_FROM:
        return None
    L = pd.concat([pd.read_csv(p) for p in C.RETAINED_FROM.split(os.pathsep)], ignore_index=True)
    return L.groupby("Stage")["Metric"].apply(list).to_dict()


if __name__ == "__main__":
    ensure(C.PCA_DIR)
    main, house = load_summaries()
    pinned = pinned_features()
    loadings, variance, scores, audit = [], [], [], []

    for g in C.PCA_GROUPS:
        data = group_data(g, main, house)
        sub = data[(data["Stage"] == g) & (data["Setup"] == "DUR")].reset_index(drop=True)
        cand = [f"{k}_{C.PCA_STAT}" for k in C.METRIC_ORDER
                if f"{k}_{C.PCA_STAT}" in sub.columns]
        cand = [c for c in cand if sub[c].notna().sum() >= C.MIN_N]
        if sub.empty or len(cand) < 3:
            print(f"  [skip] {g}: n={len(sub)}, candidates={len(cand)}")
            continue
        if pinned and g in pinned:
            cols, a = [c for c in pinned[g] if c in sub.columns], []
            print(f"  {g}: {len(cols)} pinned features (RETAINED_FROM)")
        else:
            cols, a = select_features(sub, cand, g)
        audit += a
        if len(cols) < 3:
            print(f"  [skip] {g}: {len(cols)} features after filtering")
            continue

        Z = StandardScaler().fit_transform(impute(sub, cols).values)
        nc = min(C.N_PCS, Z.shape[0] - 1, Z.shape[1])
        pca = PCA(n_components=nc, random_state=C.SEED).fit(Z)
        ev = pca.explained_variance_ratio_
        print(f"  {g}: n={len(sub)} features={len(cols)} "
              + " ".join(f"PC{i + 1}={v * 100:.1f}%" for i, v in enumerate(ev[:4])))

        L = pd.DataFrame({"Stage": g, "Metric": cols, "Category": [category(c) for c in cols]})
        for i in range(nc):
            L[f"PC{i + 1}_loading"] = pca.components_[i]
        loadings.append(L)
        variance.append(pd.DataFrame({"Stage": g, "PC": [f"PC{i + 1}" for i in range(nc)],
                                      "Variance": ev, "Cumulative": np.cumsum(ev)}))
        S = pd.DataFrame(pca.transform(Z), columns=[f"PC{i + 1}" for i in range(nc)])
        scores.append(pd.concat([sub[C.KEYS + ["noise_type", "treatment"]], S], axis=1))

    save_csv(pd.concat(loadings, ignore_index=True), "pca_loadings_by_stage", C.PCA_DIR)
    save_csv(pd.concat(variance, ignore_index=True), "pca_variance_by_stage", C.PCA_DIR)
    save_csv(pd.concat(scores, ignore_index=True), "pca_scores_by_stage", C.PCA_DIR)
    save_csv(pd.DataFrame(audit), "pca_feature_filter_audit", C.PCA_DIR)
    print("Done - 03_feature_selection_pca")
