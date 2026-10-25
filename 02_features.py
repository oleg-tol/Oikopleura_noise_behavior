"""
02_features.py - trajectory metrics and per-animal feature tables.

1. Kinematic and arena metrics from the 6 Hz centre-of-mass tables (wide, animal x part).
2. Parts stitched per animal; per-animal mean / p50 / p90 of every metric.
3. Adult DUR noise window split into inside / outside the house (manual entry/exit times).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import ensure, meta_cols, save_csv, time_cols

COORDS_DIR  = os.path.join(C.LOAD_DIR, "coordinates")
SKEL_DIR    = os.path.join(C.LOAD_DIR, "skel_metrics")
METRICS_DIR = os.path.join(C.FEAT_DIR, "metrics")


# =============================================================================
# 1. TRAJECTORY METRICS
# =============================================================================
def like(template, values):
    return pd.concat([template[meta_cols(template)].reset_index(drop=True),
                      pd.DataFrame(values, columns=time_cols(template))], axis=1)


def prepend_nan(a):
    return np.insert(a, 0, np.nan, axis=1)


def kinematics(com_x, com_y):
    tc = time_cols(com_x)
    x, y = com_x[tc].to_numpy(float), com_y[tc].to_numpy(float)
    dx, dy = np.diff(x, axis=1), np.diff(y, axis=1)
    speed = prepend_nan(np.hypot(dx, dy) * C.PX_TO_MM / C.DT_RS)
    accel = prepend_nan(np.abs(np.diff(speed, axis=1)) / C.DT_RS)
    heading = prepend_nan(np.arctan2(dy, dx))
    dh = (np.diff(heading, axis=1) + np.pi) % (2 * np.pi) - np.pi
    omega = prepend_nan(np.degrees(dh) / C.DT_RS)
    ang_acc = prepend_nan(np.diff(omega, axis=1) / C.DT_RS)
    return {"speed_wide": speed, "acceleration_wide": accel, "heading_wide": heading,
            "omega_signed_from_com_wide": omega, "omega_abs_from_com_wide": np.abs(omega),
            "angular_acceleration_signed_wide": ang_acc,
            "angular_acceleration_abs_wide": np.abs(ang_acc)}


def arena(com_x, com_y, w=C.WIN_RS_FRAMES):
    """Sliding-window (3 s) path-shape metrics; value assigned to the window start."""
    tc = time_cols(com_x)
    x, y = com_x[tc].to_numpy(float), com_y[tc].to_numpy(float)
    n_rows, n_time = x.shape
    xw, yw = sliding_window_view(x, w, axis=1), sliding_window_view(y, w, axis=1)
    fin = np.isfinite(xw) & np.isfinite(yw)
    allf, enough = fin.all(-1), fin.sum(-1) >= 4
    sx, sy = np.diff(xw, axis=-1), np.diff(yw, axis=-1)
    disp = np.hypot(xw[..., -1] - xw[..., 0], yw[..., -1] - yw[..., 0])
    path = np.nansum(np.hypot(sx, sy), axis=-1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dx = xw - np.nanmean(xw, -1, keepdims=True)
        dy = yw - np.nanmean(yw, -1, keepdims=True)
        r = np.hypot(dx, dy)
        vx, vy, cxy = np.nanvar(dx, -1), np.nanvar(dy, -1), np.nanmean(dx * dy, -1)
        tr, det = vx + vy, vx * vy - cxy ** 2
        disc = np.sqrt(np.maximum(tr ** 2 - 4 * det, 0))
        l1, l2 = (tr + disc) / 2, (tr - disc) / 2
        theta = np.unwrap(np.arctan2(sy, sx), axis=-1)
        p = np.maximum(np.stack([l1, l2], -1), 0)
        p = p / np.where((s := p.sum(-1, keepdims=True)) > 0, s, 1)
        out = {
            "net_displacement_wide": np.where(allf, disp, np.nan),
            "tortuosity_wide": np.where(allf & (disp >= 5) & (path > 0), path / disp, np.nan),
            "r_gyr_wide": np.where(enough, np.sqrt(np.nanmean(r ** 2, -1)), np.nan),
            "ringness_wide": np.where(enough & (np.nanmean(r, -1) > 0),
                                      np.nanstd(r, -1) / np.nanmean(r, -1), np.nan),
            "anisotropy_wide": np.where(enough & (l2 > 0), l1 / l2, np.nan),
            "winding_wide": np.where(allf, np.abs(theta[..., -1] - theta[..., 0]) / (2 * np.pi), np.nan),
            "path_entropy_wide": np.where(allf & (np.nanstd(xw, -1) > 0) & (np.nanstd(yw, -1) > 0),
                                          -np.sum(p * np.log2(p + 1e-10), -1), np.nan),
        }
    full = {}
    for k, v in out.items():
        a = np.full((n_rows, n_time), np.nan)
        a[:, :v.shape[1]] = v
        full[k] = a
    return full


# =============================================================================
# 2. STITCHING AND PER-ANIMAL SUMMARY
# =============================================================================
def load_metric_tables():
    com_x = pd.read_csv(os.path.join(COORDS_DIR, "com_x_wide.csv"), low_memory=False)
    com_y = pd.read_csv(os.path.join(COORDS_DIR, "com_y_wide.csv"), low_memory=False)
    assert meta_cols(com_x) == meta_cols(com_y) and len(com_x) == len(com_y)
    tables = {k: like(com_x, v) for k, v in {**kinematics(com_x, com_y), **arena(com_x, com_y)}.items()}
    for name, df in tables.items():
        save_csv(df, name, METRICS_DIR)
    for _, stem, _, cat in C.METRICS:
        if cat == "skeleton":
            tables[stem] = pd.read_csv(os.path.join(SKEL_DIR, f"{stem}.csv"), low_memory=False)
    return {key: tables[stem] for key, stem, _, _ in C.METRICS}


def stitch(df, T):
    """Animal x (N_PARTS * T) array; each part goes into its own slot (missing part = NaN)."""
    df = df.copy()
    df[C.KEYS] = df[C.KEYS].fillna("unknown")
    df = df[df["Part"].between(1, C.N_PARTS)]
    keys = df[C.KEYS].drop_duplicates().reset_index(drop=True)
    pos = {k: i for i, k in enumerate(keys.itertuples(index=False, name=None))}
    vals = df[time_cols(df)[:T]].to_numpy(float)
    arr = np.full((len(keys), C.N_PARTS * T), np.nan)
    for r, (k, p) in enumerate(zip(df[C.KEYS].itertuples(index=False, name=None), df["Part"])):
        sl = slice((int(p) - 1) * T, int(p) * T)
        if np.isnan(arr[pos[k], sl]).all():          # duplicate part: first one kept
            arr[pos[k], sl] = vals[r]
    return keys, arr


def stats_of(arr, prefix):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {f"{prefix}_mean": np.nanmean(arr, 1), f"{prefix}_p50": np.nanmedian(arr, 1),
                f"{prefix}_p90": np.nanpercentile(arr, 90, axis=1),
                f"{prefix}_nan_frac": np.isnan(arr).mean(1)}


def per_animal_summary(stitched):
    out = None
    for key, (keys, arr) in stitched.items():
        df = pd.concat([keys, pd.DataFrame(stats_of(arr, key))], axis=1)
        out = df if out is None else out.merge(df, on=C.KEYS, how="outer")
    return out


# =============================================================================
# 3. ADULT HOUSE SPLIT (DUR, noise window)
# =============================================================================
def house_masks(T):
    h = pd.read_excel(C.HOUSE_XLSX).rename(columns={"condition": "Condition", "setup": "Setup"})
    h = h[h["Setup"] == "DUR"]
    to_f = lambda t: int(round(t * 60 / C.DT_RS))
    masks = {}
    for _, r in h.iterrows():
        inside = np.full(T, C.DEFAULT_HOUSE_STATE == "inside")
        if bool(r.get("entry_observed")) and pd.notna(r.get("entry_time_min_concat")):
            e = min(max(to_f(r["entry_time_min_concat"]), 0), T)
            inside[:e], inside[e:] = False, True
            if bool(r.get("exit_observed")) and pd.notna(r.get("exit_time_min_concat")):
                x = min(max(to_f(r["exit_time_min_concat"]), 0), T)
                if x > e:
                    inside[x:] = False
        masks[(r["Condition"], r["unique_animal"])] = inside
    return masks


def house_summary(stitched, T):
    masks = house_masks(T)
    sl = slice((C.HOUSE_NOISE_PART - 1) * T, C.HOUSE_NOISE_PART * T)
    rows, audit = [], []
    for (cond, uid), inside in masks.items():
        audit.append({"Condition": cond, "unique_animal": uid,
                      "n_inside_frames": int(inside.sum()), "n_outside_frames": int((~inside).sum())})
        for state, m in (("inside", inside), ("outside", ~inside)):
            if m.sum() < C.MIN_FRAMES_PER_SUBSET:
                continue
            rec = {"Stage": f"adult_{state}_house", "Condition": cond, "Setup": "DUR", "unique_animal": uid}
            for key, (keys, arr) in stitched.items():
                i = np.flatnonzero((keys["Stage"] == "adult") & (keys["Setup"] == "DUR")
                                   & (keys["Condition"] == cond) & (keys["unique_animal"] == uid))
                if i.size == 0:
                    continue
                v = arr[i[0], sl][m]
                v = v[np.isfinite(v)]
                if v.size >= C.MIN_FRAMES_PER_SUBSET:
                    rec.update({f"{key}_mean": v.mean(), f"{key}_p50": np.median(v),
                                f"{key}_p90": np.percentile(v, 90)})
            rows.append(rec)
    audit = pd.DataFrame(audit)
    print(f"  house table animals: {len(audit)} | >= {C.MIN_FRAMES_PER_SUBSET} frames inside: "
          f"{(audit['n_inside_frames'] >= C.MIN_FRAMES_PER_SUBSET).sum()}, outside: "
          f"{(audit['n_outside_frames'] >= C.MIN_FRAMES_PER_SUBSET).sum()}")
    return pd.DataFrame(rows), audit


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    ensure(METRICS_DIR)
    print("=== 1. Trajectory metrics ===")
    tables = load_metric_tables()

    print("\n=== 2. Per-animal summary ===")
    T = min(len(time_cols(df)) for df in tables.values())
    print(f"  frames per part: {T} ({T * C.DT_RS / 60:.1f} min)")
    stitched = {k: stitch(df, T) for k, df in tables.items()}
    summary = per_animal_summary(stitched)
    save_csv(summary, "per_animal_summary", C.FEAT_DIR)
    print(summary.groupby(["Stage", "Setup", "Condition"]).size().to_string())

    print("\n=== 3. Adult house split ===")
    if os.path.exists(C.HOUSE_XLSX):
        hs, audit = house_summary(stitched, T)
        save_csv(hs, "adult_house_split_summary", C.FEAT_DIR)
        save_csv(audit, "adult_house_mask_audit", C.FEAT_DIR)
    else:
        print(f"  [skip] house table not found: {C.HOUSE_XLSX}")
    print("Done - 02_features")
