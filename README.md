# paper_xgb_colsample (reproduction package)

This repository rebuilds all tables/figures for `paper/main_v3.tex` and (optionally) regenerates the underlying simulation results from scratch.

`paper/main_v3.pdf` is included as a convenience build output; it can be regenerated locally.

### arXiv link: https://arxiv.org/abs/2601.08121

## Quickstart (rebuild tables/figures from shipped aggregates)

1) Install Python dependencies:

- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

2) Rebuild paper assets (tables + figures):

- `python scripts/build_paper_assets.py`

3) Compile the paper PDF:

- `cd paper && pdflatex main_v3.tex && pdflatex main_v3.tex`

This path does not require retraining models; it uses the shipped aggregate CSVs in `outputs/aggregate/*`.

## Optional: regenerate aggregates from scratch

This reruns the full simulation/training pipeline for the exact configurations used to produce the shipped aggregates.

- `python scripts/rebuild_aggregates_from_scratch.py`

Then rebuild assets + compile as above.

## Repo layout (high level)

- `paper/`: LaTeX source (`main_v3.tex`) plus generated inputs (`assets/`, `tables/`).
- `outputs/aggregate/*`: shipped aggregates used to rebuild the paper quickly.
- `configs/`: exact configs used for the from-scratch regeneration.
- `xgb_intratree_colsamp_ratios/`: minimal Python package (DGPs + training + aggregation).
- `scripts/`: entry points (`build_paper_assets.py`, `rebuild_aggregates_from_scratch.py`).
