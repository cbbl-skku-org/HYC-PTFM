"""Input validation for the 15 predictors and checks used for warnings."""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import config as cfg


class ValidationError(ValueError):
    """Raised when a value cannot be read as a number."""


def normalize_column(name: str) -> str:
    """'C (wt%)' -> 'C', 'Temp (°C)' -> 'Temp'; names without a unit are returned unchanged."""
    return re.sub(r"\s*\(.*\)\s*$", "", str(name)).strip()


def _to_float(value: Any) -> float:
    """Convert a user value to float; empty values become NaN."""
    if value is None:
        return float("nan")
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
        if value == "" or value.lower() in {"na", "nan", "null", "none", "-"}:
            return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"'{value}' is not a number")
    if math.isinf(out):
        raise ValidationError("infinite values are not allowed")
    return out


def clean_records(records: Iterable[dict]) -> tuple[pd.DataFrame, list[list[str]]]:
    """Read records into a frame with the 15 predictor columns and return per-record blocking errors."""
    rows, errors = [], []
    for rec in records:
        rec = {normalize_column(k): v for k, v in rec.items()} if isinstance(rec, dict) else {}
        err: list[str] = []
        row = {}
        for key in cfg.INPUT_KEYS:
            try:
                row[key] = _to_float(rec.get(key))
            except ValidationError as exc:
                row[key] = float("nan")
                err.append(f"{key}: {exc}")
        for key in cfg.REQUIRED_KEYS:
            if math.isnan(row[key]) and not any(e.startswith(f"{key}:") for e in err):
                err.append(f"{key}: required value is missing")
        for key, value in row.items():
            if not math.isnan(value) and value < 0:
                err.append(f"{key}: must not be negative")
        if not math.isnan(row["FC"]) and row["FC"] <= 0:
            err.append("FC: must be greater than 0")
        if not math.isnan(row["C"]) and row["C"] <= 0:
            err.append("C: must be greater than 0")
        rows.append(row)
        errors.append(err)
    return pd.DataFrame(rows, columns=cfg.INPUT_KEYS), errors


def quality_warnings(frame: pd.DataFrame, ranges: pd.DataFrame) -> list[list[str]]:
    """Non-blocking warnings per record."""
    out: list[list[str]] = []
    lo_u, hi_u = cfg.ULTIMATE_SUM_RANGE
    lo_p, hi_p = cfg.PROXIMATE_SUM_RANGE
    for i in range(len(frame)):
        r = frame.iloc[i]
        w: list[str] = []
        ult = np.nansum([r["C"], r["H"], r["O"], r["N"], r["S"]])
        if not lo_u <= ult <= hi_u:
            w.append(f"C + H + O + N + S = {ult:.1f} wt% (expected {lo_u:.0f}-{hi_u:.0f}); check units and basis")
        if not (np.isnan(r["VM/FC"]) or np.isnan(r["FC"])):
            prox = r["VM/FC"] * r["FC"] + r["FC"] + r["Ash"]
            if not lo_p <= prox <= hi_p:
                w.append(f"VM + FC + Ash = {prox:.1f} wt% (expected {lo_p:.0f}-{hi_p:.0f}); check VM/FC, FC and Ash")
        hc = r["H"] / (r["C"] / 12.0)
        if abs(hc - r["H/C"]) > 0.05 * max(hc, 1e-9):
            w.append(f"H/C = {r['H/C']:.3g} differs from (H/1)/(C/12) = {hc:.3g}; check how H/C was calculated")
        missing = [k for k in cfg.OPTIONAL_KEYS if np.isnan(r[k])]
        if missing:
            w.append("Imputed from the training data: " + ", ".join(missing)
                     + ". Predictions are less reliable when these values are missing.")
        outside = []
        for name in cfg.MODEL_FEATURES:
            v = r[name]
            if not np.isnan(v) and (v < ranges.loc[name, "min"] or v > ranges.loc[name, "max"]):
                outside.append(f"{name} = {v:.4g} (training range {ranges.loc[name, 'min']:.4g}-{ranges.loc[name, 'max']:.4g})")
        if outside:
            w.append("Outside the range of the training data, so the prediction is an extrapolation: " + "; ".join(outside))
        out.append(w)
    return out
