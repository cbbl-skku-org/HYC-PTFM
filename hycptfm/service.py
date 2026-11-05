"""High-level prediction service shared by the command line and the web server."""
from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from . import config as cfg
from .features import clean_records, normalize_column, quality_warnings
from .model import HYCPTFM


def _num(x: float, digits: int = 3):
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else round(float(x), digits)


class PredictionService:
    def __init__(self, model: HYCPTFM | None = None):
        self.model = model or HYCPTFM.fit()

    def schema(self) -> dict:
        return {
            "inputs": [f.__dict__ for f in cfg.INPUT_FIELDS],
            "targets": [t.__dict__ for t in cfg.TARGETS],
            "interval": {"lower_quantile": cfg.INTERVAL_QUANTILES[0], "upper_quantile": cfg.INTERVAL_QUANTILES[1]},
            "training": {"records": self.model.n_train,
                         "biomass_types": 117,
                         "splits": {"repeats": 20, "train_fraction": 0.8, "seeds": [46, 65],
                                    "stratification": "10 quantile bins of the target"},
                         "ranges": {k: {"min": _num(v["min"], 4), "max": _num(v["max"], 4)}
                                    for k, v in self.model.ranges.iterrows()}},
            "external_validation": cfg.EXTERNAL_VALIDATION,
            "scope": cfg.SCOPE_NOTICE,
            "limits": {"max_rows": cfg.MAX_ROWS, "max_upload_bytes": cfg.MAX_UPLOAD_BYTES,
                       "requests_per_minute": cfg.RATE_LIMIT_PER_MIN},
            "backend": self.model.backend,
        }

    def predict(self, records: Iterable[dict], targets: list[str] | None = None) -> list[dict]:
        """Predict the selected targets (default: all) for each record.

        One dict per record: {"index", "ok", "errors", "warnings", "predictions"}. Invalid records carry
        errors and no predictions; user values never raise.
        """
        targets = [t for t in (targets or cfg.TARGET_KEYS) if t in cfg.TARGET_KEYS] or cfg.TARGET_KEYS
        records = list(records)
        frame, errors = clean_records(records)
        results = [{"index": i, "ok": not errors[i], "errors": errors[i], "warnings": [], "predictions": None}
                   for i in range(len(records))]
        ok_idx = [i for i, e in enumerate(errors) if not e]
        if not ok_idx:
            return results
        good = frame.iloc[ok_idx]
        warns = quality_warnings(good, self.model.ranges)
        pred = self.model.predict(good, targets)
        units = {t.key: t.unit for t in cfg.TARGETS}
        for pos, i in enumerate(ok_idx):
            results[i]["warnings"] = warns[pos]
            results[i]["predictions"] = {
                k: {"value": _num(pred[k].iloc[pos]["value"]), "lower": _num(pred[k].iloc[pos]["lower"]),
                    "upper": _num(pred[k].iloc[pos]["upper"]), "unit": units[k]}
                for k in targets}
        return results

    def predict_frame(self, df: pd.DataFrame, targets: list[str] | None = None) -> pd.DataFrame:
        """Batch helper: the input table plus prediction, interval, warnings and error columns."""
        df = df.rename(columns={c: normalize_column(c) for c in df.columns})
        return self.to_table(df, self.predict(df.to_dict(orient="records"), targets), targets)

    @staticmethod
    def to_table(df: pd.DataFrame, res: list[dict], targets: list[str] | None = None) -> pd.DataFrame:
        out = df.copy().reset_index(drop=True)
        keys = [k for k in cfg.TARGET_KEYS if not targets or k in targets] or cfg.TARGET_KEYS
        for k in keys:
            for part, name in (("value", f"{k}_pred"), ("lower", f"{k}_lower"), ("upper", f"{k}_upper")):
                out[name] = [r["predictions"][k][part] if r["ok"] and r["predictions"] else None for r in res]
        out["warnings"] = [" | ".join(r["warnings"]) for r in res]
        out["errors"] = [" | ".join(r["errors"]) for r in res]
        return out
