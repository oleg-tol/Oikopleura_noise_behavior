"""
01_loading.py - DLC CSVs -> filtered centre-of-mass trajectories and skeleton metrics.

Parses stage/noise type/setup/part from folder and file names (with a naming
validator), filters keypoints (likelihood, IQR, step size, short runs), interpolates,
resamples 30 -> 6 Hz and writes wide tables (one row per animal x part).
"""
import os
import re
import sys
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import label as nd_label
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import ensure, save_csv, time_cols

COORDS_DIR = os.path.join(C.LOAD_DIR, "coordinates")
SKEL_DIR   = os.path.join(C.LOAD_DIR, "skel_metrics")
META_DIR   = os.path.join(C.LOAD_DIR, "metadata")
N_JOBS     = -4

FOLDER_META_MAP = {   # folder -> (stage, noise type, setup)
    "ADULTS_BOAT_AF": ("adult", "boat", "AF"),         "ADULTS_BOAT_DUR": ("adult", "boat", "DUR"),
    "ADULTS_IG_AF": ("adult", "artificial", "AF"),     "ADULTS_IG_DUR": ("adult", "artificial", "DUR"),
    "JUVENAILS_BOAT_AF": ("juvenile", "boat", "AF"),   "JUVENAILS_BOAT_DUR": ("juvenile", "boat", "DUR"),
    "JUVENAILS_IG_AF": ("juvenile", "artificial", "AF"), "JUVENAILS_IG_DUR": ("juvenile", "artificial", "DUR"),
    "LARVAE_BOAT_AF": ("larva", "boat", "AF"),         "LARVAE_BOAT_DUR": ("larva", "boat", "DUR"),
    "LARVAE_IG_AF": ("larva", "artificial", "AF"),     "LARVAE_IG_DUR": ("larva", "artificial", "DUR"),
}
_STAGE = r"(Adults?|Juveniles?|Larvae?|dults?|uveniles?|arvae?)"
RE_FULL   = re.compile(rf"^(\d{{8}})_(\d{{6}})_+{_STAGE}_+(Ig|b|BOAT)?(NOISE|CONTROL|BOATNOISE)_+(\d+)[-_](\d+)DLC", re.I)
RE_NOPART = re.compile(rf"^(\d{{8}})_(\d{{6}})_+{_STAGE}_+(Ig|b|BOAT)?(NOISE|CONTROL|BOATNOISE)_+(\d+)DLC", re.I)
RE_NOSTG  = re.compile(r"^(\d{8})_(\d{6})_+\w*_*(?:Ig|b|BOAT)?(NOISE|CONTROL|BOATNOISE)_+(\d+)[-_](\d+)DLC", re.I)


# =============================================================================
# FILES AND METADATA
# =============================================================================
def collect_csvs(root):
    """All DLC CSVs; if '_filtered' / '_el' / plain versions coexist, keep the best."""
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in sorted(fs)
             if f.lower().endswith(".csv") and not f.startswith("._")]
    prio = lambda p: 0 if p.lower().endswith("_filtered.csv") else 1 if p.lower().endswith("_el.csv") else 2
    stem = lambda p: re.sub(r"_(filtered|el)\.csv$", ".csv", p.lower())
    groups = {}
    for p in files:
        groups.setdefault(stem(p), []).append(p)
    kept = [p for g in groups.values() for p in g if prio(p) == min(map(prio, g))]
    print(f"[collect] {len(files)} CSVs -> {len(kept)} after removing superseded versions")
    return sorted(kept)


def norm_stage(raw, fallback):
    if raw is None:
        return fallback
    s = raw.lower()
    return ("adult" if s.startswith(("adult", "dult")) else
            "juvenile" if s.startswith(("juv", "uven")) else
            "larva" if s.startswith(("larv", "arva")) else fallback)


def norm_noise(pfx, treat, fallback):
    pfx, treat = (pfx or "").lower(), treat.lower()
    noise = "artificial" if "ig" in pfx else "boat" if pfx in ("b", "boat") or "boat" in treat else None
    return noise or fallback, noise is None


def epoch_from_part(stage, setup, treatment, part):
    """(Phase, Noise_Epoch, Stimulus_Applied). DUR adults: noise starts at part 1."""
    if part is None:
        return "unknown", "unknown", False
    if treatment != "noise":
        return f"control_part_{part}", "control", False
    if setup == "DUR" and stage == "adult":
        return ("noise", "during_noise", True) if part == 1 else ("after_noise", "after_noise", False)
    return {1: ("before_noise", "before_noise", False),
            2: ("noise", "during_noise", True),
            3: ("after_noise", "after_noise", False)}.get(part, ("unknown", "unknown", False))


def parse_metadata(fp):
    fn = os.path.basename(fp).lstrip("._")
    folder = os.path.basename(os.path.dirname(fp)).upper().strip()
    f_stage, f_noise, f_setup = FOLDER_META_MAP.get(folder, (None, None, None))
    meta = dict(file=os.path.basename(fp), folder=folder, setup=f_setup, regex="failed", warnings=[],
                stage=f_stage, noise_type=f_noise, treatment=None, date=None, animal_id=None,
                part=None, is_long_file=False)
    if folder not in FOLDER_META_MAP:
        meta["warnings"].append(f"UNKNOWN_FOLDER '{folder}'")

    if (m := RE_FULL.match(fn)):
        date, ts, stg, pfx, treat, aid, part = m.groups(); meta["regex"] = "full"
    elif (m := RE_NOPART.match(fn)):
        date, ts, stg, pfx, treat, aid = m.groups(); part = None
        meta.update(regex="no_part", is_long_file=True)
        meta["warnings"].append("LONG_FILE: auto-split into 3 parts")
    elif (m := RE_NOSTG.match(fn)):
        date, ts, treat, aid, part = m.groups(); stg = pfx = None; meta["regex"] = "lenient"
        meta["warnings"].append("LENIENT_MATCH: stage from folder")
    else:
        meta["warnings"].append("REGEX_FAIL")
        return meta

    stage = norm_stage(stg, f_stage)
    noise, from_folder = norm_noise(pfx, treat, f_noise)
    if stg and f_stage and stage != f_stage:
        meta["warnings"].append(f"STAGE_MISMATCH file={stage} folder={f_stage}")
    if not from_folder and f_noise and noise != f_noise:
        meta["warnings"].append(f"NOISE_CONFLICT file={noise} folder={f_noise}")
    for v, fmt in ((date, "%Y%m%d"), (ts, "%H%M%S")):
        try:
            datetime.strptime(v, fmt)
        except ValueError:
            meta["warnings"].append(f"BAD_DATE/TIME '{v}'")
    meta.update(stage=stage, noise_type=noise, date=date, animal_id=aid,
                treatment="noise" if "noise" in treat.lower() else "control",
                part=int(part) if part else None)
    if meta["part"] not in (None, 1, 2, 3):
        meta["warnings"].append(f"UNEXPECTED_PART {meta['part']}")
    return meta


# =============================================================================
# FILTERING
# =============================================================================
def read_dlc(fp):
    hdr = pd.read_csv(fp, nrows=3, header=None)
    indiv = hdr.iloc[1, 1::3].fillna("Unknown").astype(str).tolist()
    raw = pd.read_csv(fp, header=None, skiprows=3, low_memory=False).apply(pd.to_numeric, errors="coerce")
    return indiv, *(raw.iloc[:, k::3].to_numpy(float, copy=True) for k in (1, 2, 3))


def iqr_filter(a, thr=C.IQR_THRESHOLD):
    q25, q75 = np.nanpercentile(a, 25, axis=0), np.nanpercentile(a, 75, axis=0)
    out = a.copy()
    out[(out < q25 - thr * (q75 - q25)) | (out > q75 + thr * (q75 - q25))] = np.nan
    return out


def step_filter(x, y, thr=C.DISTANCE_THRESHOLD):
    d = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
    bad = np.vstack([np.zeros((1, x.shape[1]), bool), ~(d <= thr)])
    x, y = x.copy(), y.copy()
    x[bad] = y[bad] = np.nan
    return x, y


def mask_short_runs(a, min_len):
    if min_len <= 1:
        return a
    lab, n = nd_label(np.isfinite(a), structure=np.array([[0, 1, 0]] * 3, bool))
    if n == 0:
        return a
    short = np.flatnonzero(np.bincount(lab.ravel(), minlength=n + 1)[1:] < min_len) + 1
    out = a.copy()
    out[np.isin(lab, short)] = np.nan
    return out


def interpolate_inside(a):
    """Linear interpolation of internal gaps only (no extrapolation)."""
    out, idx = a.astype(float), np.arange(a.shape[0])
    for j in range(out.shape[1]):
        v = np.isfinite(out[:, j])
        if v.sum() < 2 or v.all():
            continue
        f, l = np.argmax(v), len(v) - 1 - np.argmax(v[::-1])
        out[f:l + 1, j] = np.interp(idx[f:l + 1], idx[v], out[v, j])
    return out


def nanrow_mean(a):
    with np.errstate(all="ignore"):
        return np.nanmean(a, axis=1)


# =============================================================================
# SKELETON METRICS
# =============================================================================
def skeleton_metrics(x, y):
    """Curvature, mean-centred tangent angle and quirkiness over keypoints 0 + tail (>=4)."""
    T, n = x.shape
    out = {k: np.full(T, np.nan) for k in ("curv", "tan", "quir")}
    idx = np.r_[0, np.arange(4, n)] if n > 4 else np.array([0])
    if idx.size < 3:
        return out
    xs, ys = x[:, idx], y[:, idx]
    w = min(C.SAVGOL_WL, idx.size if idx.size % 2 else idx.size - 1)
    kw = dict(polyorder=C.SAVGOL_ORDER, mode="nearest", axis=1)
    x0, y0 = np.nan_to_num(xs), np.nan_to_num(ys)
    dx, dy = savgol_filter(x0, w, deriv=1, **kw), savgol_filter(y0, w, deriv=1, **kw)
    ddx, ddy = savgol_filter(x0, w, deriv=2, **kw), savgol_filter(y0, w, deriv=2, **kw)
    curv = np.abs((dx[:, :-2] * ddy[:, 2:] - dy[:, :-2] * ddx[:, 2:])
                  / (dx[:, :-2] ** 2 + dy[:, :-2] ** 2 + 1e-6) ** 1.5)
    out["curv"] = nanrow_mean(np.where(curv <= 0.2, curv, np.nan))
    ang = np.arctan2(np.diff(ys, axis=1), np.diff(xs, axis=1))
    mu = np.nanmean(ang, axis=1) if np.isfinite(ang).any() else np.zeros(T)
    out["tan"] = nanrow_mean((ang - np.nan_to_num(mu)[:, None]) % (2 * np.pi))
    k = min(10, idx.size)
    axis = np.arctan2(np.nanmean(ys[:, -k:], 1) - np.nanmean(ys[:, :k], 1),
                      np.nanmean(xs[:, -k:], 1) - np.nanmean(xs[:, :k], 1))
    out["quir"] = np.nansum(np.abs((ang - axis[:, None] + np.pi) % (2 * np.pi) - np.pi), axis=1)
    return out


def tail_lateral(x, y):
    """Lateral displacement of the tail tip relative to the trunk axis (native rate)."""
    n = x.shape[1]
    if n <= 4:
        return np.full(x.shape[0], np.nan)
    base = np.stack([x[:, 0], y[:, 0]], -1)
    trunk = np.stack([np.nanmean(x[:, 1:4], 1), np.nanmean(y[:, 1:4], 1)], -1) - base
    u = trunk / np.where((nrm := np.linalg.norm(trunk, axis=-1, keepdims=True)) == 0, np.nan, nrm)
    normal = np.stack([-u[:, 1], u[:, 0]], -1)
    return np.sum((np.stack([x[:, -1], y[:, -1]], -1) - base) * normal, -1)


def tailbeat(lat_nv, T_rs):
    """Dominant frequency (Hann FFT, 3 s) and half peak-to-peak amplitude (1 s), per 6 Hz frame."""
    freq, amp = np.full(T_rs, np.nan), np.full(T_rs, np.nan)
    end = np.arange(T_rs) * C.RESAMPLE_RATE
    ok = (end - C.WIN_NV_FRAMES + 1 >= 0) & (end < lat_nv.size)
    if not ok.any():
        return freq, amp
    k = np.flatnonzero(ok)
    W = lat_nv[(end[ok] - C.WIN_NV_FRAMES + 1)[:, None] + np.arange(C.WIN_NV_FRAMES)]
    mag = np.abs(np.fft.rfft(W * np.hanning(C.WIN_NV_FRAMES), axis=1))[:, 1:]
    freq[k] = np.fft.rfftfreq(C.WIN_NV_FRAMES, 1 / C.FPS_NATIVE)[np.argmax(mag, 1) + 1]
    f1 = int(round(C.FPS_NATIVE))
    ok1 = end[ok] - f1 + 1 >= 0
    W1 = lat_nv[(end[ok][ok1] - f1 + 1)[:, None] + np.arange(f1)]
    with np.errstate(all="ignore"):
        amp[k[ok1]] = (np.nanmax(W1, 1) - np.nanmin(W1, 1)) / 2
    return freq, amp


# =============================================================================
# PER FILE
# =============================================================================
KEYS_RS = ["com_x", "com_y", "curv", "tan", "quir", "tb_freq", "tb_amp"]
KEYS_NV = ["com_x_native", "com_y_native"]


def process_file(fp, meta):
    stage, setup = meta["stage"], meta["setup"] or "unknown"
    if stage is None:
        print(f"[SKIP] no stage: {meta['file']}")
        return None
    cond = f"{meta['noise_type'] or 'unknown'}_{meta['treatment'] or 'unknown'}"
    exp_id = os.path.splitext(meta["file"])[0]
    warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows are expected
    try:
        indiv, cx, cy, lk = read_dlc(fp)
        cx[lk < C.LIKELIHOOD_MIN] = cy[lk < C.LIKELIHOOD_MIN] = np.nan
        cx, cy = step_filter(iqr_filter(cx), iqr_filter(cy))
        cx = interpolate_inside(mask_short_runs(cx, C.MIN_RUN_NATIVE_FRAMES.get(stage, 0)))
        cy = interpolate_inside(mask_short_runs(cy, C.MIN_RUN_NATIVE_FRAMES.get(stage, 0)))
        max_rs = int(C.MAX_PART_SEC * C.FPS_NATIVE / C.RESAMPLE_RATE)
        if meta["is_long_file"]:
            max_rs *= C.LONG_FILE_N_PARTS
        cx, cy = cx[:max_rs * C.RESAMPLE_RATE], cy[:max_rs * C.RESAMPLE_RATE]
        rx, ry = cx[::C.RESAMPLE_RATE], cy[::C.RESAMPLE_RATE]

        if meta["is_long_file"]:   # no part number: split into equal thirds
            Tr, Tn = rx.shape[0] // C.LONG_FILE_N_PARTS, cx.shape[0] // C.LONG_FILE_N_PARTS
            last = C.LONG_FILE_N_PARTS - 1
            segs = [(p + 1, slice(p * Tr, None if p == last else (p + 1) * Tr),
                     slice(p * Tn, None if p == last else (p + 1) * Tn))
                    for p in range(C.LONG_FILE_N_PARTS)]
        else:
            segs = [(meta["part"], slice(None), slice(None))]

        labels = np.array(indiv)
        out = {k: [] for k in KEYS_RS + KEYS_NV}
        summary = []
        for part, srs, snv in segs:
            phase, epoch, stim = epoch_from_part(stage, setup, meta["treatment"], part)
            for ind in sorted(set(labels)):
                j = np.flatnonzero(labels == ind)
                xr, yr, xn, yn = rx[srs][:, j], ry[srs][:, j], cx[snv][:, j], cy[snv][:, j]
                if not np.isfinite(xr).any():
                    continue
                trunk = np.arange(min(4, len(j)))
                sk = skeleton_metrics(xr, yr)
                f, a = tailbeat(tail_lateral(xn, yn), xr.shape[0])
                vals = {"com_x": nanrow_mean(xr[:, trunk]), "com_y": nanrow_mean(yr[:, trunk]),
                        "curv": sk["curv"], "tan": sk["tan"], "quir": sk["quir"],
                        "tb_freq": f, "tb_amp": a,
                        "com_x_native": nanrow_mean(xn[:, trunk]), "com_y_native": nanrow_mean(yn[:, trunk])}
                row = {"Experiment_ID": exp_id, "Stage": stage, "Condition": cond, "Setup": setup,
                       "Individual": ind, "Part": part, "Phase": phase, "Noise_Epoch": epoch,
                       "Stimulus_Applied": stim,
                       "Segment_Source": "auto_split_long_file" if meta["is_long_file"] else "filename_part",
                       "unique_animal": f"{meta['date']}_{meta['animal_id']}_{ind}"}
                for k, v in vals.items():
                    out[k].append((row, v))
                summary.append({**row, "T_native": xn.shape[0], "T_resampled": xr.shape[0],
                                "N_keypoints": len(j), "is_long_file": meta["is_long_file"]})
        return {**{k: v for k, v in out.items()}, "summary": summary}
    except Exception as exc:
        print(f"  [WARN] skipped {meta['file']}: {exc}")
        traceback.print_exc()
        return None


def to_wide(pairs):
    """[(meta_dict, 1-D array)] -> wide DataFrame padded with NaN to the longest row."""
    if not pairs:
        return pd.DataFrame()
    T = max(len(v) for _, v in pairs)
    arr = np.full((len(pairs), T), np.nan)
    for i, (_, v) in enumerate(pairs):
        arr[i, :len(v)] = v
    return pd.concat([pd.DataFrame([m for m, _ in pairs]),
                      pd.DataFrame(arr, columns=[f"Time_{i:03d}" for i in range(T)])], axis=1)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    ensure(COORDS_DIR, SKEL_DIR, META_DIR)
    files = collect_csvs(C.DATA_DIR)
    metas = {fp: parse_metadata(fp) for fp in files}

    diag = pd.DataFrame([{**{k: v for k, v in m.items() if k != "warnings"},
                          "n_warnings": len(m["warnings"]), "warnings": " | ".join(m["warnings"])}
                         for m in metas.values()])
    save_csv(diag, "naming_diag", META_DIR)
    print(diag["regex"].value_counts().to_string())
    n_conf = diag["warnings"].str.contains("CONFLICT|MISMATCH|REGEX_FAIL").sum()
    if n_conf:
        print(f"[WARN] {n_conf} files with naming conflicts/failures - check naming_diag.csv")

    print(f"\n=== Processing {len(files)} files ===")
    res = [r for r in Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(process_file)(fp, metas[fp]) for fp in files) if r is not None]
    print(f"processed {len(res)} / {len(files)}")

    wide = {k: to_wide([p for r in res for p in r[k]]) for k in KEYS_RS + KEYS_NV}
    assert len(wide["com_x"]) == len(wide["com_y"]), "com_x / com_y misaligned"
    names = {"com_x": "com_x_wide", "com_y": "com_y_wide",
             "com_x_native": "com_x_native_wide", "com_y_native": "com_y_native_wide",
             "curv": "curv_wide", "tan": "tan_angles_wide", "quir": "quirkiness_wide",
             "tb_freq": "tailbeat_freq_wide", "tb_amp": "tailbeat_amp_wide"}
    for k, df in wide.items():
        save_csv(df, names[k], COORDS_DIR if k.startswith("com") else SKEL_DIR)
    save_csv(pd.DataFrame([s for r in res for s in r["summary"]]), "skeleton_summary", META_DIR)
    print("Done - 01_loading")
