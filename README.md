# NFL Learning Tool

A baseline pipeline to scrape historical NFL game results from Pro-Football-Reference (PFR), create features (rolling averages + Elo), train regressors to predict weekly game scores, and output predictions for upcoming games.

## Files
- `src/data/scraper.py` - Scrapes PFR "years/<year>/games.htm" and caches results.
- `src/data/processor.py` - Builds features: rolling averages, rest days, Elo.
- `src/models/train.py` - Trains two regressors (home/away points) and saves models.
- `src/models/predict.py` - Predicts scores for a given year/week using saved models.
- `src/utils/elo.py` - Elo implementation.
- `pyproject.toml` - dependencies

## Quick start
1. Create virtualenv and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
