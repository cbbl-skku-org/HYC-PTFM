"""Static configuration of HYC-PTFM: feature definitions, targets and model settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELING_TABLE = str(DATA_DIR / "modeling" / "{target}_full.csv")   # 15 predictors + target, as used in the study
IMPUTATION_TABLE = DATA_DIR / "cleaned_1020_with_missing.csv"      # 1,020 records with the missing values left empty
CASE_STUDY_TABLE = DATA_DIR / "case_study" / "Yield_case_study.csv"   # 141 external yield records (11 predictors)

RANDOM_STATE = 46            # seed of the study (repeated splits use seeds 46-65)
IMPUTER_SEED = 42            # seed of the ERT imputation of the study
TABPFN_N_ESTIMATORS = 8      # value selected in 16-18 of the 20 repeated splits for each target


@dataclass(frozen=True)
class Field:
    key: str        # column name in the dataset and in API payloads
    label: str
    unit: str
    required: bool
    help: str


# The 15 predictors the user supplies, in the order used in the study (H/C and VM/FC are entered directly).
INPUT_FIELDS = [
    Field("C", "Carbon", "wt%", True, "Ultimate analysis of the feedstock"),
    Field("H", "Hydrogen", "wt%", True, "Ultimate analysis of the feedstock"),
    Field("O", "Oxygen", "wt%", True, "Ultimate analysis of the feedstock"),
    Field("N", "Nitrogen", "wt%", True, "Ultimate analysis of the feedstock"),
    Field("S", "Sulfur", "wt%", False, "Optional; imputed when empty"),
    Field("H/C", "H/C atomic ratio", "-", True, "(H / 1) / (C / 12), with H and C in wt%"),
    Field("FC", "Fixed carbon", "wt%", True, "Proximate analysis"),
    Field("Ash", "Ash", "wt%", True, "Proximate analysis"),
    Field("VM/FC", "VM/FC ratio", "-", True, "Volatile matter divided by fixed carbon (wt% / wt%)"),
    Field("CL", "Cellulose", "wt%", False, "Optional; imputed when empty"),
    Field("HC", "Hemicellulose", "wt%", False, "Optional; imputed when empty"),
    Field("LG", "Lignin", "wt%", False, "Optional; imputed when empty"),
    Field("Temp", "Pyrolysis temperature", "°C", True, "Final (peak) temperature"),
    Field("Time", "Residence time", "min", False, "Optional; imputed when empty"),
    Field("HR", "Heating rate", "°C/min", False, "Optional; imputed when empty"),
]
INPUT_KEYS = [f.key for f in INPUT_FIELDS]
REQUIRED_KEYS = [f.key for f in INPUT_FIELDS if f.required]
OPTIONAL_KEYS = [f.key for f in INPUT_FIELDS if not f.required]
MODEL_FEATURES = INPUT_KEYS


@dataclass(frozen=True)
class Target:
    key: str
    label: str
    unit: str
    # Testing performance of HYC-PTFM, mean +/- SD over the 20 repeated 80:20 splits (Table S4)
    test_r2: float
    test_r2_sd: float
    test_rmse: float
    test_rmse_sd: float
    test_mae: float
    test_mae_sd: float
    cv_r2: float          # training-set 10-fold CV R2, mean over the same 20 splits
    lower_bound: float = 0.0
    upper_bound: float | None = None


# Values of Table S4 (mean +/- SD over the 20 repeated target-stratified 80:20 splits, seeds 46-65).
TARGETS = [
    Target("HHV", "Higher heating value of the biochar", "MJ/kg",
           0.913, 0.033, 1.142, 0.210, 0.628, 0.073, 0.904),
    Target("Yield", "Biochar yield", "%",
           0.967, 0.011, 4.539, 0.699, 2.676, 0.277, 0.962, 0.0, 100.0),
    Target("CER", "Char energy recovery rate", "%",
           0.939, 0.018, 6.324, 0.871, 3.756, 0.347, 0.932, 0.0, None),
]
TARGET_KEYS = [t.key for t in TARGETS]

# External validation of biochar yield on the 141 case-study records with the 11 shared predictors (Table S6).
EXTERNAL_VALIDATION = {
    "target": "Yield",
    "records": 141,
    "biomass_types": 34,
    "predictors": 11,
    "r2": 0.802, "r2_sd": 0.022,
    "rmse": 5.721, "rmse_sd": 0.319,
    "mae": 4.053, "mae_sd": 0.287,
    "internal_r2_same_predictors": 0.966,
    "bias_at_yield_ge_50_pp": -6.77,
    "n_records_yield_ge_50": 21,
}

# Scope of the released models. Returned by /api/schema and shown on the web page; keep in step with the article.
SCOPE_NOTICE = (
    "HYC-PTFM is intended for exploratory screening of feedstocks and pyrolysis conditions within the range of "
    "the compiled dataset (1,020 literature records, 117 biomass types). Its predictions are statistical "
    "associations learned from heterogeneous literature data, not thermochemical mechanisms, and are not a "
    "substitute for measurement. Yield was validated on 141 independent records (R2 = 0.802); HHV and CER have "
    "not been validated externally. Treat inputs flagged as outside the training range as extrapolation."
)

# Prediction interval reported by the server (TabPFN predictive quantiles)
INTERVAL_QUANTILES = (0.05, 0.95)

# Plausibility checks used for warnings (same limits as the data-quality flags of the study)
ULTIMATE_SUM_RANGE = (90.0, 103.0)    # C + H + O + N + S, wt%
PROXIMATE_SUM_RANGE = (90.0, 105.0)   # VM + FC + Ash, wt% (VM = VM/FC x FC)

# Runtime settings (environment variables)
BACKEND = os.environ.get("HYC_BACKEND", "tabpfn")        # "tabpfn" (default) or "stub" (tests only)
DEVICE = os.environ.get("HYC_DEVICE", "cpu")             # "cpu" or "cuda"
MAX_ROWS = int(os.environ.get("HYC_MAX_ROWS", "500"))    # records per request
MAX_UPLOAD_BYTES = int(os.environ.get("HYC_MAX_UPLOAD_BYTES", str(1_000_000)))
RATE_LIMIT_PER_MIN = int(os.environ.get("HYC_RATE_LIMIT_PER_MIN", "30"))
