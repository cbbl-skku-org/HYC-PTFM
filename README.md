# HYC-PTFM

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%E2%80%933.11-blue.svg)](https://www.python.org/)
[![Web server](https://img.shields.io/badge/web%20server-balalab--skku.org-brightgreen.svg)](https://balalab-skku.org/HYC-PTFM/)

**Predict biochar higher heating value (HHV), yield and char energy recovery rate (CER) from feedstock
composition and pyrolysis conditions, using a pretrained tabular foundation model (TabPFN).**

Install this repository to run predictions on your own data offline. For a few samples, use the web server
instead: **https://balalab-skku.org/HYC-PTFM/**.

HYC-PTFM is one TabPFN regressor per target, fitted on 1,020 curated literature records (117 biomass types).
TabPFN trains no weights, so there is no model file to download: fitting gives the pretrained checkpoint those
records as in-context examples, which takes about 10 s at each start.

> **Intended use: exploratory screening within the range of the compiled data** — narrowing a candidate list or
> prioritising experiments. It is a statistical model, not a thermochemical one, and is no substitute for
> measurement. Read [Limitations](#limitations) before relying on a prediction.

## Installation

Python 3.9-3.11, about 2 GB of memory, CPU only. Internet is needed on the first run to fetch the TabPFN
checkpoint, which is then cached.

```bash
conda create -n hycptfm python=3.9 -y && conda activate hycptfm
git clone https://github.com/Cobonla/HYC-PTFM.git && cd HYC-PTFM
pip install -r requirements.txt
```

## Usage

```bash
python predictor.py --input_file <INPUT.csv> --output_file <OUTPUT.csv>
```

| Option | Meaning |
|---|---|
| `--input_file` | your input table (CSV) |
| `--output_file` | where to write the predictions (default: `result.csv`) |
| `--targets` | any subset of `HHV Yield CER` (default: all three) |
| `--template [PATH]` | write an example input table to start from (default: `my_data.csv`) and exit |

```bash
python predictor.py --input_file examples/example.csv --output_file result.csv   # try it
python predictor.py --template my_data.csv                                      # start your own table
```

A run takes about 6 s per target for up to 50 rows and about 27 s for 500 rows with all three targets, because
the cost is dominated by the fixed 1,020-record context rather than by your row count.

## Input

A CSV table with a header row and one row per sample.
[`examples/example.csv`](examples/example.csv) is a complete table you can copy.

| Column | Meaning | Unit | Required |
|---|---|---|---|
| `C`, `H`, `O`, `N` | feedstock ultimate analysis | wt% | **yes** |
| `S` | sulfur | wt% | no |
| `H/C` | atomic ratio, `(H/1)/(C/12)` with H and C in wt% | – | **yes** |
| `FC`, `Ash` | fixed carbon and ash (proximate analysis) | wt% | **yes** |
| `VM/FC` | volatile matter divided by fixed carbon | – | **yes** |
| `CL`, `HC`, `LG` | cellulose, hemicellulose, lignin | wt% | no |
| `Temp` | final (peak) pyrolysis temperature | °C | **yes** |
| `Time` | residence time | min | no |
| `HR` | heating rate | °C min⁻¹ | no |

A unit in brackets after a name (`C (wt%)`) is accepted. Comma-, semicolon- and tab-separated files are all read.

**Enter composition on the basis you measured.** The training records keep the basis their source studies
reported (as received, dry, or dry-ash-free), which was deliberately not harmonised, so a different basis shifts
the prediction.

### Missing values

**The six optional columns may be left empty** — keep the column in the header and leave the cell blank
(`NA` and `nan` also count as missing). HYC-PTFM fills them with the Extra Trees imputer of the study,
`IterativeImputer(ExtraTreesRegressor(n_estimators=50, max_depth=8, random_state=42), max_iter=5,
random_state=42)`, and names every imputed column in the output `warnings`. Values you supply are never altered.
See [`examples/example_with_missing.csv`](examples/example_with_missing.csv).

The nine required columns cannot be imputed; a row missing one of them is rejected. Imputation makes a prediction
less reliable, so supply a measured value whenever you have one.

## Output

Your input table plus, per requested target, `*_pred` (MJ kg⁻¹ for HHV, % for yield and CER) and `*_lower` /
`*_upper` (the 5th and 95th percentile of TabPFN's predictive distribution), then two columns to read first:

- **`warnings`** — the row *was* predicted, but: a value was imputed; `C+H+O+N+S` is outside 90-103 wt% or
  `VM+FC+Ash` outside 90-105 wt%; the given `H/C` disagrees with `H` and `C` by more than 5 %; or a value is
  outside the training range, making the prediction an **extrapolation**.
- **`errors`** — the row was **not** predicted: a required value is missing, unreadable or negative, or `C` or
  `FC` is zero.

## Performance

Mean ± SD over 20 repeated target-stratified 80:20 splits of the 1,020 records (seeds 46-65):

| Target | Testing R² | Testing RMSE | Testing MAE |
|---|---|---|---|
| HHV (MJ kg⁻¹) | 0.913 ± 0.033 | 1.142 ± 0.210 | 0.628 ± 0.073 |
| Yield (%) | 0.966 ± 0.011 | 4.539 ± 0.699 | 2.676 ± 0.277 |
| CER (%) | 0.939 ± 0.017 | 6.324 ± 0.871 | 3.756 ± 0.347 |

Those SDs describe how much the result depends on which records land in the testing subset, not uncertainty about
a wider population of biomass. **Biochar yield was also validated externally** on 141 independent records from 34
non-overlapping biomass types, using the 11 shared predictors: R² = 0.802 ± 0.022 (RMSE 5.72 %), against
0.966 ± 0.012 internally with the same predictors, with yields of 50 % or more underpredicted by 6.77 percentage
points on average.

## Limitations

- **Predictions are statistical associations, not thermochemistry**, learned from 1,020 heterogeneous literature
  records, and carry no causal claim.
- **Only yield is validated externally. HHV and CER are not** — the external dataset contains no HHV or CER
  measurements.
- **CER is derived, not measured**: feedstock HHV is estimated with the Dulong equation and
  CER = HHV_biochar × yield / HHV_feedstock. 97 of the 1,020 values exceed 100 % (maximum 138.12 %) and were
  kept, so CER carries more uncertainty than the two measured targets.
- **Coverage is uneven.** Median temperature 300 °C, with 55.5 % of records at or below 300 °C and 10.2 % above
  600 °C; median residence time 30 min; 55.3 % at a heating rate of 10 °C min⁻¹. Expect larger errors at high
  temperature, long residence time and fast heating.
- **Training values were imputed** (S in 102 records, CL in 325, HC in 325, LG in 309, Time in 108, HR in 76),
  and the imputer was fitted on all 1,020 records before the splits, so testing-record predictors entered both
  steps. The effect on the estimates above was not quantified.
- **The intervals are not safety margins.** They describe the model's own uncertainty on data like the training
  data; their coverage on unseen feedstocks was not tested and they do not widen reliably outside the range.
- **The released predictor is a refit** on all 1,020 records, while every number above was measured on models
  holding the 816 records of a training subset. Same algorithm, hyperparameters and seed, but the accuracy of the
  released fit was not measured separately.
- **Out of scope**: quoting outputs as experimental results; regulatory, commercial, safety or process-design
  decisions; mechanistic interpretation; and processes outside slow pyrolysis of lignocellulosic and related
  biomass, such as hydrothermal carbonisation, torrefaction under steam, catalytic and co-pyrolysis.

## Repository structure

```
predictor.py      entry point
hycptfm/          input checks, ERT imputation, the three TabPFN models
data/modeling/    the 1,020 records the models are fitted on          <- required
data/cleaned_1020_with_missing.csv   the imputer is fitted on this    <- required
data/case_study/, data/imputation/   study data, not read by the tool
examples/         example input tables, with and without missing values
```

## Citation
Please cite our stydy. 
Please also cite TabPFN (Hollmann et al., *Nature* 2025;637:319-26) and the
source studies of any records you reuse.

## Licence and contact
Code: MIT ([`LICENSE`](LICENSE)). Data: compiled from published studies, redistributed with their references for
research use. TabPFN and its checkpoint carry their own licences.
