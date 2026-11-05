"""Command line interface. The documented entry point is predictor.py in the repository root;
this module is the same code reachable as a module, and is also what the web server imports.

    python -m hycptfm template  > example.csv
    python -m hycptfm predict --input example.csv --output predictions.csv
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import config as cfg

EXAMPLE_ROWS = [
    # illustrative lignocellulosic feedstocks inside the training range; H/C = (H/1)/(C/12), VM/FC = VM/FC
    dict(C=47.0, H=6.0, O=44.0, N=0.6, S=0.1, **{"H/C": 1.532}, FC=15.0, Ash=5.0, **{"VM/FC": 5.333},
         CL=40.0, HC=22.0, LG=24.0, Temp=300, Time=30, HR=10),
    dict(C=48.0, H=6.1, O=43.0, N=0.5, S=0.1, **{"H/C": 1.525}, FC=16.0, Ash=6.0, **{"VM/FC": 4.875},
         CL=38.0, HC=23.0, LG=25.0, Temp=500, Time=60, HR=10),
    dict(C=46.0, H=5.9, O=45.0, N=1.0, S=None, **{"H/C": 1.539}, FC=17.0, Ash=7.0, **{"VM/FC": 4.471},
         CL=None, HC=None, LG=None, Temp=700, Time=None, HR=None),
]


def example_frame() -> pd.DataFrame:
    return pd.DataFrame(EXAMPLE_ROWS, columns=cfg.INPUT_KEYS)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hycptfm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("template", help="print an example input table (CSV) to standard output")
    pr = sub.add_parser("predict", help="predict HHV, yield and CER for a CSV table")
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", required=True)
    args = p.parse_args(argv)

    if args.cmd == "template":
        example_frame().to_csv(sys.stdout, index=False)
        return 0
    if args.cmd == "predict":
        from .service import PredictionService
        df = pd.read_csv(args.input)
        out = PredictionService().predict_frame(df)
        out.to_csv(args.output, index=False)
        bad = int((out["errors"] != "").sum())
        print(f"{len(out)} records written to {args.output}; {bad} with errors")
        return 0 if bad == 0 else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
