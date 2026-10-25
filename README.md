# Oikopleura noise behaviour

Behavioural analysis of *Oikopleura dioica* larvae, juveniles and adults exposed to
vessel noise (Vessel-DNS) or artificial noise (MSFD-NB), recorded during (DUR) or after
(AF) exposure. DeepLabCut keypoints are converted into kinematic, arena and skeleton
features, filtered for redundancy, summarised with per-stage PCA and compared with each
group's own control.

## Modules

| Script | Input | Output | Content |
|---|---|---|---|
| `01_loading.py` | DLC CSVs | `01_loading/` | Metadata from folder/file names (with naming validator), keypoint filtering, interpolation, 30 -> 6 Hz resampling, centre of mass, skeleton metrics |
| `02_features.py` | `01_loading/` | `02_features/` | Kinematic and arena metrics, per-animal mean / p50 / p90, adult inside / outside house split |
| `03_feature_selection_pca.py` | `02_features/` | `03_selection_pca/` | NaN QC, CV filter, category-aware correlation filter, per-stage PCA |
| `04_plotting.py` | `01`-`03` | `04_figures/` | % change bubble plots, AF box plots, PCA scree and loading wheels, trajectories |
| `05_supplement_stats.py` | `02`-`03` | `05_supplement/` | Statistics tables per figure, centroid shift (permutation), Cliff's delta overview |

`config.py` holds paths, parameters, metric definitions and colours; `utils.py` holds shared helpers.

## Pipeline

```
DLC CSVs ──> 01_loading ──> 02_features ──> 03_feature_selection_pca ──> 04_plotting
                                  │                     │
                                  └─────────────────────┴──────────────> 05_supplement_stats
```

Run the scripts in order (e.g. in Spyder, from the repository folder). Set the data and
output locations in `config.py` or with environment variables:

```
NOISE_DATA_DIR    folder with the DLC CSV subfolders (e.g. LARVAE_BOAT_DUR)
NOISE_OUT_DIR     output folder (default: <NOISE_DATA_DIR>/pipeline_out)
NOISE_HOUSE_XLSX  manual house entry / exit table (adults, DUR)
NOISE_RETAINED_FROM  optional pca_loadings_by_stage.csv to pin the retained features
```

## Key parameters

| Parameter | Value | Where |
|---|---|---|
| Native / analysis frame rate | 30 Hz / 6 Hz | `FPS_NATIVE`, `RESAMPLE_RATE` |
| Keypoint likelihood threshold | 0.80 | `LIKELIHOOD_MIN` |
| Outlier filters | 2 x IQR; step > 100 px; runs < 3 s (larvae, juveniles) | `IQR_THRESHOLD`, `DISTANCE_THRESHOLD`, `MIN_RUN_NATIVE_FRAMES` |
| Sliding window (tail beat, arena metrics) | 3 s | `WIN_SEC` |
| Per-animal summaries | mean, median, 90th percentile | `STATS` |
| Feature filter | NaN <= 30 %, CV >= 0.20, Spearman abs r < 0.75 within / 0.92 across categories | `NAN_FRAC_MAX`, `CV_MIN`, `CORR_MAX`, `CORR_MAX_CROSS` |
| PCA | DUR, per-animal mean, z-scored, 6 PCs, per stage | `PCA_STAT`, `N_PCS` |
| House split | noise window, state before first entry = outside, >= 5 s per state | `DEFAULT_HOUSE_STATE`, `MIN_FRAMES_PER_SUBSET` |
| Tests | Mann-Whitney U, Cliff's delta (2000 bootstraps), centroid distance (10000 permutations) | `N_BOOT`, `N_PERM` |
| Multiple comparisons | Benjamini-Hochberg per stage x setup x noise type | `FAMILY` |

Every comparison is noise vs the control of the same stage, setup and noise type.

## Requirements

Python 3.11 with numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, joblib and
openpyxl (`pip install -r requirements.txt`).
