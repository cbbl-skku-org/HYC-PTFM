"""HYC-PTFM models: one TabPFN regressor per target and an ERT imputer for optional inputs.

The regressors are refitted on the 1,020 curated records (data/modeling) when the service starts. TabPFN does
not train network weights; "fitting" stores the training records as the in-context examples, so this takes a
few seconds and needs no model files besides the TabPFN checkpoint itself.

Imputation follows the study (Biochar_preprocessing notebook): IterativeImputer with ExtraTreesRegressor
(50 trees, maximum depth 8), 5 iterations, seed 42, fitted on all numeric columns of the cleaned table
(including the three targets and the biomass code). For a new record the targets and the biomass code are
unknown and are left empty, and the derived columns (O/C, (O+N)/C, VM, feedstock HHV) are computed.
"""
from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg

log = logging.getLogger("hycptfm")

# columns of the table used for imputation (all columns except Biomass and Reference, plus the biomass code)
_IMP_COLS = ["C (wt%)", "H (wt%)", "O (wt%)", "N (wt%)", "S (wt%)", "H/C (wt%)", "O/C (wt%)", "(O+N)/C (wt%)",
             "BHHV (MJ/kg)", "VM (wt%)", "FC (wt%)", "Ash (wt%)", "VM/FC (wt%)", "CL (wt%)", "HC (wt%)", "LG (wt%)",
             "Temp (°C)", "Time (min)", "HR (°C/min)", "HHV (MJ/kg)", "Yield (%)", "CER (%)", "Biomass_code"]
_TABLE_NAMES = {"C": "C (wt%)", "H": "H (wt%)", "O": "O (wt%)", "N": "N (wt%)", "S": "S (wt%)", "H/C": "H/C (wt%)",
                "FC": "FC (wt%)", "Ash": "Ash (wt%)", "VM/FC": "VM/FC (wt%)", "CL": "CL (wt%)", "HC": "HC (wt%)",
                "LG": "LG (wt%)", "Temp": "Temp (°C)", "Time": "Time (min)", "HR": "HR (°C/min)"}


def _imputation_input(feats: pd.DataFrame) -> pd.DataFrame:
    """Expand the 15 predictors of new records to the columns of the imputation table."""
    t = pd.DataFrame({_TABLE_NAMES[k]: feats[k].values for k in cfg.INPUT_KEYS}, index=feats.index)
    t["O/C (wt%)"] = (feats["O"] / 16.0) / (feats["C"] / 12.0)
    t["(O+N)/C (wt%)"] = (feats["O"] / 16.0 + feats["N"] / 14.0) / (feats["C"] / 12.0)
    t["BHHV (MJ/kg)"] = 0.338 * feats["C"] + 1.428 * (feats["H"] - feats["O"] / 8.0) + 0.095 * feats["S"]  # Dulong, Eq. (1)
    t["VM (wt%)"] = feats["VM/FC"] * feats["FC"]
    for c in ("HHV (MJ/kg)", "Yield (%)", "CER (%)", "Biomass_code"):
        t[c] = np.nan
    return t[_IMP_COLS]


@dataclass
class _StubRegressor:
    """Mean predictor used only for tests and continuous integration (no TabPFN weights needed)."""
    mean: float = 0.0
    sd: float = 1.0

    def predict(self, X, output_type: str = "mean", quantiles=None):
        n = len(X)
        if output_type == "quantiles":
            from scipy.stats import norm
            return [np.full(n, self.mean + self.sd * norm.ppf(q)) for q in quantiles]
        return np.full(n, self.mean)


@dataclass
class HYCPTFM:
    imputer: object = None
    regressors: dict = field(default_factory=dict)
    ranges: pd.DataFrame = None
    n_train: int = 0
    backend: str = "tabpfn"

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, backend: str | None = None, device: str | None = None) -> "HYCPTFM":
        backend = backend or cfg.BACKEND
        device = device or cfg.DEVICE

        tables = {t.key: pd.read_csv(cfg.MODELING_TABLE.format(target=t.key)) for t in cfg.TARGETS}
        X0 = tables["HHV"].iloc[:, :15].astype(float)
        X0.columns = cfg.MODEL_FEATURES
        ranges = pd.DataFrame({"min": X0.min(), "max": X0.max()})

        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.preprocessing import LabelEncoder
        raw = pd.read_csv(cfg.IMPUTATION_TABLE)
        raw["Biomass_code"] = LabelEncoder().fit_transform(raw["Biomass"].astype(str))
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=50, max_depth=8, random_state=cfg.IMPUTER_SEED, n_jobs=1),
            max_iter=5, random_state=cfg.IMPUTER_SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputer.fit(raw[_IMP_COLS].astype(float))

        regressors = {}
        for t in cfg.TARGETS:
            df = tables[t.key]
            X, y = df.iloc[:, :15].astype(float).values, df.iloc[:, -1].astype(float).values
            if backend == "stub":
                regressors[t.key] = _StubRegressor(float(y.mean()), float(y.std()))
                continue
            # The final models use all 1,020 records; TabPFN refuses more than 1,000 samples on a CPU
            # unless this is allowed explicitly (inference is slower but works).
            os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
            from tabpfn import TabPFNRegressor
            reg = TabPFNRegressor(device=device, random_state=cfg.RANDOM_STATE, n_estimators=cfg.TABPFN_N_ESTIMATORS)
            reg.fit(X, y)
            regressors[t.key] = reg
            log.info("fitted TabPFN for %s on %d records", t.key, len(y))
        return cls(imputer=imputer, regressors=regressors, ranges=ranges, n_train=len(X0), backend=backend)

    # --------------------------------------------------------------- prediction
    def impute(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Fill empty optional values (S, CL, HC, LG, Time, HR); other values are returned unchanged."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr = self.imputer.transform(_imputation_input(feats).astype(float))
        full = pd.DataFrame(arr, columns=_IMP_COLS, index=feats.index)
        out = feats.copy()
        for k in cfg.OPTIONAL_KEYS:
            out[k] = feats[k].where(feats[k].notna(), full[_TABLE_NAMES[k]])
        return out[cfg.MODEL_FEATURES]

    def predict(self, feats: pd.DataFrame, targets=None) -> dict[str, pd.DataFrame]:
        """Return {target: DataFrame(value, lower, upper)} for the (possibly incomplete) predictors."""
        X = self.impute(feats).values
        lo_q, hi_q = cfg.INTERVAL_QUANTILES
        out = {}
        for t in cfg.TARGETS:
            if targets and t.key not in targets:
                continue
            reg = self.regressors[t.key]
            mean = np.asarray(reg.predict(X), dtype=float)
            try:
                q = reg.predict(X, output_type="quantiles", quantiles=[lo_q, hi_q])
                lower, upper = np.asarray(q[0], float), np.asarray(q[1], float)
            except Exception:  # interval not available for this TabPFN version
                lower = upper = np.full(len(X), np.nan)
            res = pd.DataFrame({"value": mean, "lower": lower, "upper": upper}).clip(lower=t.lower_bound)
            if t.upper_bound is not None:
                res = res.clip(upper=t.upper_bound)
            out[t.key] = res
        return out
