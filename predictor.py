"""HYC-PTFM: predict biochar HHV, yield and char energy recovery rate (CER) for your own data.

    python predictor.py --input_file examples/example.csv --output_file result.csv

The input is a CSV table with a header row and one row per sample (see examples/). The six optional
predictors (S, CL, HC, LG, Time, HR) may be left empty; they are then filled with the Extra Trees
imputer of the study and reported in the "warnings" column. Run with --template to write a starting file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from hycptfm import config as cfg
from hycptfm.cli import example_frame

HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="predictor.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_file", help="input table (CSV) with the 15 predictors")
    p.add_argument("--output_file", default="result.csv", help="where to write the predictions (default: result.csv)")
    p.add_argument("--targets", nargs="+", choices=cfg.TARGET_KEYS, default=cfg.TARGET_KEYS,
                   metavar="TARGET", help=f"subset of {' '.join(cfg.TARGET_KEYS)} (default: all three)")
    p.add_argument("--template", metavar="PATH", nargs="?", const="my_data.csv",
                   help="write an example input table to PATH (default: my_data.csv) and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.template:
        example_frame().to_csv(args.template, index=False)
        print(f"Example input table written to {args.template}. Replace the rows with your own samples.")
        return 0

    if not args.input_file:
        print("error: --input_file is required (or use --template to create a starting file)", file=sys.stderr)
        return 2

    try:
        df = pd.read_csv(args.input_file, sep=None, engine="python")   # comma, semicolon or tab
    except Exception as exc:
        print(f"error: {args.input_file} is not a readable CSV table ({exc})", file=sys.stderr)
        return 2
    if df.empty:
        print(f"error: {args.input_file} has a header but no rows", file=sys.stderr)
        return 2

    print(f"Read {len(df)} samples from {args.input_file}.")
    print("Fitting the three TabPFN models on the 1,020 curated records (about 10 s, no weights are trained)...")

    from hycptfm.service import PredictionService
    out = PredictionService().predict_frame(df, args.targets)
    out.to_csv(args.output_file, index=False)

    n_err = int((out["errors"] != "").sum())
    n_warn = int((out["warnings"] != "").sum())
    print(f"\n{len(out)} rows written to {args.output_file} "
          f"({len(out) - n_err} predicted, {n_err} rejected, {n_warn} with warnings).")
    if n_warn:
        print('Read the "warnings" column: it flags imputed values, implausible composition sums '
              "and inputs outside the training range (extrapolation).")
    if n_err:
        print('Read the "errors" column: those rows received no prediction.')
        for i, msg in out.loc[out["errors"] != "", "errors"].head(5).items():
            print(f"  row {i + 2}: {msg}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
