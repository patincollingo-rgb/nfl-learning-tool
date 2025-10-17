# src/models/train.py
import argparse
import pandas as pd
import numpy as np
import os
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from src.data.scraper import get_games_df_for_year
from src.data.processor import make_features_from_games

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def gather_years(start_year, end_year, force=False):
    frames = []
    for y in range(start_year, end_year + 1):
        df = get_games_df_for_year(y, force=force)
        # only keep rows where we successfully parsed home_abbr and visitor_abbr
        if "home_abbr" not in df.columns or "visitor_abbr" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        raise RuntimeError("No data collected")
    all_games = pd.concat(frames, ignore_index=True)
    return all_games

def train_models(all_games_df):
    feats = make_features_from_games(all_games_df, rolling_window=3)
    # only rows with actual scores for training
    train_df = feats.dropna(subset=["home_points","away_points"]).copy()
    # target: home_points and away_points
    y_home = train_df["home_points"].astype(float)
    y_away = train_df["away_points"].astype(float)
    feature_cols = [
        "home_elo","away_elo","elo_diff",
        "home_avg_pf_last3","home_avg_pa_last3","away_avg_pf_last3","away_avg_pa_last3",
        "home_games_played","away_games_played",
        "home_rest_days","away_rest_days",
        "pf_diff_last3"
    ]
    X = train_df[feature_cols].fillna(0).values
    # simple RandomForest regressors
    model_home = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model_away = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    print("Training home points model on", X.shape[0], "rows")
    model_home.fit(X, y_home)
    print("Training away points model on", X.shape[0], "rows")
    model_away.fit(X, y_away)
    # persist
    dump(model_home, os.path.join(MODEL_DIR, "model_home.joblib"))
    dump(model_away, os.path.join(MODEL_DIR, "model_away.joblib"))
    # Save feature columns to file
    pd.Series(feature_cols).to_csv(os.path.join(MODEL_DIR, "feature_cols.csv"), index=False, header=False)
    print("Models saved to", MODEL_DIR)
    return model_home, model_away, feature_cols

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    all_games = gather_years(args.start_year, args.end_year, force=args.force)
    train_models(all_games)
