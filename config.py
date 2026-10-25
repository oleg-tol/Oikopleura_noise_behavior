"""
config.py - shared paths, recording parameters, metric definitions and colours.
Paths can be overridden with environment variables (NOISE_DATA_DIR, NOISE_OUT_DIR,
NOISE_HOUSE_XLSX, NOISE_RETAINED_FROM).
"""
import os

# =============================================================================
# PATHS
# =============================================================================
DATA_DIR   = os.environ.get("NOISE_DATA_DIR", "/path/to/DLC_csv_root")
OUT_DIR    = os.environ.get("NOISE_OUT_DIR", os.path.join(DATA_DIR, "pipeline_out"))
HOUSE_XLSX = os.environ.get("NOISE_HOUSE_XLSX",
                            os.path.join(DATA_DIR, "manual_house_entry_with_exit_timepoints.xlsx"))
# Optional: pin retained features to an existing pca_loadings_by_stage.csv
# (skips the automatic filter; used to reproduce frozen figure panels exactly).
RETAINED_FROM = os.environ.get("NOISE_RETAINED_FROM") or None

LOAD_DIR = os.path.join(OUT_DIR, "01_loading")
FEAT_DIR = os.path.join(OUT_DIR, "02_features")
PCA_DIR  = os.path.join(OUT_DIR, "03_selection_pca")
FIG_DIR  = os.path.join(OUT_DIR, "04_figures")
SUPP_DIR = os.path.join(OUT_DIR, "05_supplement")

# =============================================================================
# RECORDING / FILTERING
# =============================================================================
FPS_NATIVE    = 30.0
RESAMPLE_RATE = 5                              # 30 Hz -> 6 Hz
DT_RS         = RESAMPLE_RATE / FPS_NATIVE
WIN_SEC       = 3.0                            # sliding window (tail beat, arena metrics)
WIN_NV_FRAMES = int(round(WIN_SEC * FPS_NATIVE))
WIN_RS_FRAMES = int(round(WIN_SEC / DT_RS))
N_PARTS       = 3
MAX_PART_SEC  = 15 * 60
LONG_FILE_N_PARTS = 3

LIKELIHOOD_MIN     = 0.80
IQR_THRESHOLD      = 2.0
DISTANCE_THRESHOLD = 100.0                     # px per native frame
SAVGOL_WL, SAVGOL_ORDER = 15, 2
PIXEL_TO_UM = 11.56
PX_TO_MM    = PIXEL_TO_UM / 1000.0
MIN_RUN_NATIVE_FRAMES = {"larva": WIN_NV_FRAMES, "juvenile": WIN_NV_FRAMES, "adult": 0}

# =============================================================================
# DESIGN
# =============================================================================
STAGES      = ["larva", "juvenile", "adult"]
SETUPS      = ["DUR", "AF"]
NOISE_TYPES = ["boat", "artificial"]
NOISE_LABEL = {"boat": "Vessel-DNS", "artificial": "MSFD-NB"}
KEYS        = ["Stage", "Condition", "Setup", "unique_animal"]

# DUR part that contains the noise stimulus
NOISE_PART = {"larva": 2, "juvenile": 2, "adult": 1}

# Adult house occupancy (DUR, noise window = part 1 for adults)
DEFAULT_HOUSE_STATE   = "outside"              # state before first observed entry
HOUSE_NOISE_PART      = 1
MIN_FRAMES_PER_SUBSET = 30                     # ~5 s at 6 Hz

# Groups analysed with PCA / effect sizes (Stage value in the summary tables)
PCA_GROUPS  = ["larva", "juvenile", "adult", "adult_outside_house", "adult_inside_house"]
GROUP_LABEL = {"larva": "Larva", "juvenile": "Juvenile", "adult": "Adult (whole)",
               "adult_outside_house": "Adult outside", "adult_inside_house": "Adult inside"}

# =============================================================================
# METRICS  (key, wide-table stem, label, category)
# =============================================================================
METRICS = [
    ("speed",          "speed_wide",                         "Speed (mm/s)",                "kinematic"),
    ("omega_abs",      "omega_abs_from_com_wide",            "Abs angular velocity (deg/s)", "kinematic"),
    ("omega_signed",   "omega_signed_from_com_wide",         "Angular velocity (deg/s)",     "kinematic"),
    ("accel",          "acceleration_wide",                  "Acceleration (mm/s2)",         "kinematic"),
    ("ang_accel_abs",  "angular_acceleration_abs_wide",      "Abs angular acceleration",     "kinematic"),
    ("ang_accel_sign", "angular_acceleration_signed_wide",   "Angular acceleration",         "kinematic"),
    ("anisotropy",     "anisotropy_wide",                    "Anisotropy",                   "arena"),
    ("tortuosity",     "tortuosity_wide",                    "Tortuosity",                   "arena"),
    ("winding",        "winding_wide",                       "Winding",                      "arena"),
    ("r_gyr",          "r_gyr_wide",                         "Radius of gyration (px)",      "arena"),
    ("ringness",       "ringness_wide",                      "Ringness",                     "arena"),
    ("path_entropy",   "path_entropy_wide",                  "Path entropy",                 "arena"),
    ("heading",        "heading_wide",                       "Heading (rad)",                "arena"),
    ("net_displace",   "net_displacement_wide",              "Net displacement (px)",        "arena"),
    ("avg_curvature",  "curv_wide",                          "Avg curvature",                "skeleton"),
    ("avg_tan_angles", "tan_angles_wide",                    "Avg tangent angles",           "skeleton"),
    ("quirkiness",     "quirkiness_wide",                    "Quirkiness",                   "skeleton"),
    ("tailbeat_freq",  "tailbeat_freq_wide",                 "Tailbeat frequency (Hz)",      "skeleton"),
    ("tailbeat_amp",   "tailbeat_amp_wide",                  "Tailbeat amplitude (px)",      "skeleton"),
]
CATEGORY = {k: c for k, _, _, c in METRICS}
PRETTY   = {"speed": "Speed", "omega_abs": "Abs angular velocity", "omega_signed": "Angular velocity",
            "accel": "Acceleration", "ang_accel_abs": "Abs angular acceleration",
            "ang_accel_sign": "Angular acceleration", "anisotropy": "Anisotropy",
            "tortuosity": "Tortuosity", "winding": "Winding", "r_gyr": "Radius of gyration",
            "ringness": "Ringness", "path_entropy": "Path entropy", "heading": "Heading",
            "net_displace": "Net displacement", "avg_curvature": "Avg curvature",
            "avg_tan_angles": "Avg tangent angles", "quirkiness": "Quirkiness",
            "tailbeat_freq": "Tailbeat frequency", "tailbeat_amp": "Tailbeat amplitude"}
METRIC_ORDER = [k for k, *_ in METRICS]
STATS = ("mean", "p50", "p90")

# =============================================================================
# FIGURE CONTENT
# =============================================================================
# Bubble panels (DUR, per-animal mean, PCA-retained metrics): panel -> group
FIG_BUBBLES = {"Fig1C_larva": "larva", "Fig1G_juvenile": "juvenile",
               "Fig1J_adult_outside_house": "adult_outside_house",
               "Fig2B_adult_inside_house": "adult_inside_house"}
# Fig 3 box plots (AF, per-animal p90): panel -> (stage, noise type, metrics)
FIG3_STAT = "p90"
FIG3 = {"Fig3B_MSFD-NB_larva":       ("larva", "artificial", ["tailbeat_amp", "omega_abs", "accel"]),
        "Fig3C_MSFD-NB_juvenile":    ("juvenile", "artificial", ["tailbeat_amp"]),
        "Fig3D_Vessel-DNS_larva":    ("larva", "boat", ["speed", "r_gyr", "omega_abs", "net_displace",
                                                        "heading", "avg_curvature", "anisotropy", "accel"]),
        "Fig3E_Vessel-DNS_juvenile": ("juvenile", "boat", ["tortuosity", "tailbeat_amp",
                                                           "heading", "anisotropy"])}
SUPP_PCA_STAGES = ["larva", "juvenile", "adult"]          # Fig S1
N_TRAJ = 20                                               # example trajectories per stage x condition

# =============================================================================
# FEATURE SELECTION / PCA
# =============================================================================
PCA_STAT         = "mean"
NAN_FRAC_MAX     = 0.30
CV_MIN           = 0.20
CORR_MAX         = 0.75                        # within category
CORR_MAX_CROSS   = 0.92                        # across categories
MIN_PER_CATEGORY = {"kinematic": 3, "arena": 2, "skeleton": 2, "state": 2}
N_PCS            = 6
IMPUTE_PCA       = "per_condition"             # descriptive PCA (Fig S1); stats use pooled

# =============================================================================
# STATISTICS
# =============================================================================
SEED   = 42
N_PERM = 10000
N_BOOT = 2000
MIN_N  = 3
FAMILY = ["Stage", "Setup", "noise_type"]      # BH family: stage x setup x noise type
DELTA_VMAX = 0.6

# =============================================================================
# COLOURS / STYLE
# =============================================================================
CONTROL_COLOR = "#2980b9"
NOISE_COLOR   = {"boat": "#c0392b", "artificial": "#6c3483"}
NOISE_MARKER  = {"boat": "o", "artificial": "s"}
CAT_COLOR = {"kinematic": "#CC3311", "arena": "#0077BB", "skeleton": "#009988", "state": "#EE7733"}
CAT_BAND  = {"kinematic": "#FFF3E0", "arena": "#E8EEF4", "skeleton": "#E8F4F0", "state": "#FDEFE3"}
INCREASE_COLOR, DECREASE_COLOR = "#CC3311", "#0077BB"

RC = {"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
      "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "legend.frameon": False,
      "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
      "xtick.major.width": 0.6, "ytick.major.width": 0.6, "svg.fonttype": "none",
      "pdf.fonttype": 42}
MM = 1 / 25.4
