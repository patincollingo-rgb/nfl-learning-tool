# src/models/predict.py
import argparse
import pandas as pd
import os
from joblib import load
import numpy as np
from src.data.scraper import get_games_df_for_year
from src.data.processor import make_features_from_games

MODEL_DIR = "models"

def load_models():
    model_home = load(os.path.join(MODEL_DIR, "model_home.joblib"))
    model_away = load(os.path.join(MODEL_DIR, "model_away.joblib"))
    feature_cols = list(pd.read_csv(os.path.join(MODEL_DIR, "feature_cols.csv"), header=None).iloc[:,0].values)
    return model_home, model_away, feature_cols

def predict_week(year, week, out_csv=None):
    # gather games for year
    games_df = get_games_df_for_year(year)
    # keep only games matching the week
    week_df = games_df[games_df["week"] == week].copy()
    if week_df.empty:
        print("No games found for year", year, "week", week)
        return None
    # build features (this builds features for whole season up to each game)
    feat_df = make_features_from_games(games_df, rolling_window=3)
    # merge with week rows by home+date
    # simplify: find rows in feat_df where home==home_abbr and date==date
    merged = pd.merge(week_df[['date','home_abbr','visitor_abbr']], feat_df,
                      left_on=['date','home_abbr'],
                      right_on=['date','home'],
                      how='left',
                      suffixes=(None, "_feat"))
    model_home, model_away, feature_cols = load_models()
    # fill missing feature values with overall mean
    for c in feature_cols:
        if c not in merged.columns:
            merged[c] = 0
    X = merged[feature_cols].fillna(0).values
    pred_home = model_home.predict(X)
    pred_away = model_away.predict(X)
    merged["pred_home_points"] = np.round(pred_home).astype(int)
    merged["pred_away_points"] = np.round(pred_away).astype(int)
    merged["pred_margin"] = merged["pred_home_points"] - merged["pred_away_points"]
    out = merged[['date','home_abbr','visitor_abbr','pred_home_points','pred_away_points','pred_margin']]
    if out_csv:
        out.to_csv(out_csv, index=False)
        print("Wrote predictions to", out_csv)
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    res = predict_week(args.year, args.week, args.out)
    if res is not None:
        print(res)
