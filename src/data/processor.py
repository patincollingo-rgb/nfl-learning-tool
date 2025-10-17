# src/data/processor.py
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import timedelta
from src.utils.elo import Elo

def build_team_game_history(games_df):
    """
    Convert a year's games DataFrame into per-team chronological game records.
    Expect games_df to have columns: date, week, home_abbr, visitor_abbr, home_points, visitor_points
    """
    # Clean and sort by date (fallback to index if NaT)
    games_df = games_df.copy()
    games_df = games_df.sort_values(["date"]).reset_index(drop=True)
    # Build per-team lists
    team_games = defaultdict(list)
    for _, row in games_df.iterrows():
        if pd.isna(row.get("home_abbr")) or pd.isna(row.get("visitor_abbr")):
            continue
        game_date = row.get("date", pd.NaT)
        home = row["home_abbr"]
        away = row["visitor_abbr"]
        hpts = row.get("home_points", np.nan)
        vpts = row.get("visitor_points", np.nan)
        week = row.get("week", np.nan)
        rec = {
            "date": game_date,
            "opponent": away,
            "home": 1,
            "points_for": hpts,
            "points_against": vpts,
            "week": week,
            "opp_abbr": away
        }
        team_games[home].append(rec)
        rec2 = {
            "date": game_date,
            "opponent": home,
            "home": 0,
            "points_for": vpts,
            "points_against": hpts,
            "week": week,
            "opp_abbr": home
        }
        team_games[away].append(rec2)
    return team_games

def make_features_from_games(games_df, rolling_window=3):
    """
    Create a features DataFrame where each row is a single game, with features for home and away teams
    """
    games_df = games_df.copy()
    games_df = games_df.sort_values("date").reset_index(drop=True)
    # initialize Elo by running through games chronologically across seasons
    elo = Elo()
    features = []
    # For quick per-team last N averages, maintain running lists
    team_hist = {}
    for idx, row in games_df.iterrows():
        home = row["home_abbr"]
        away = row["visitor_abbr"]
        date = row["date"]
        week = row.get("week", None)
        # prepare team history dicts
        for t in (home, away):
            if t not in team_hist:
                team_hist[t] = {"dates": [], "pf": [], "pa": []}
        # compute last N averages (excluding current game)
        def last_stats(team):
            h = team_hist.get(team, {"pf": [], "pa": [], "dates": []})
            pf = h["pf"][-rolling_window:] if len(h["pf"])>0 else []
            pa = h["pa"][-rolling_window:] if len(h["pa"])>0 else []
            if pf:
                return {
                    "avg_pf": float(np.mean(pf)),
                    "avg_pa": float(np.mean(pa)),
                    "games_played": len(h["pf"]),
                    "rest_days": (date - h["dates"][-1]).days if h["dates"] else None
                }
            else:
                return {"avg_pf": None, "avg_pa": None, "games_played":0, "rest_days":None}
        home_stats = last_stats(home)
        away_stats = last_stats(away)
        # Elo ratings before this game (Elo stores base without home field)
        home_elo = elo.get(home)
        away_elo = elo.get(away)
        # assemble feature row
        fr = {
            "date": date,
            "week": week,
            "home": home,
            "away": away,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "home_avg_pf_last3": home_stats["avg_pf"],
            "home_avg_pa_last3": home_stats["avg_pa"],
            "home_games_played": home_stats["games_played"],
            "home_rest_days": home_stats["rest_days"],
            "away_avg_pf_last3": away_stats["avg_pf"],
            "away_avg_pa_last3": away_stats["avg_pa"],
            "away_games_played": away_stats["games_played"],
            "away_rest_days": away_stats["rest_days"],
            "home_points": row.get("home_points", None),
            "away_points": row.get("visitor_points", None),
        }
        features.append(fr)
        # After feature extraction, update histories and Elo using actual scores if available
        hpts = row.get("home_points", None)
        vpts = row.get("visitor_points", None)
        if not pd.isna(hpts) and not pd.isna(vpts):
            # update team_hist
            team_hist[home]["dates"].append(date)
            team_hist[home]["pf"].append(hpts)
            team_hist[home]["pa"].append(vpts)
            team_hist[away]["dates"].append(date)
            team_hist[away]["pf"].append(vpts)
            team_hist[away]["pa"].append(hpts)
            # update Elo
            elo.update(home, away, hpts, vpts)
    feat_df = pd.DataFrame(features)
    # Fill NA numeric with reasonable defaults (0 or league average)
    for c in ["home_avg_pf_last3","home_avg_pa_last3","away_avg_pf_last3","away_avg_pa_last3"]:
        feat_df[c] = feat_df[c].astype(float)
        feat_df[c] = feat_df[c].fillna(feat_df[c].mean())
    # rest days fill - large value if none
    feat_df["home_rest_days"] = feat_df["home_rest_days"].fillna(7)
    feat_df["away_rest_days"] = feat_df["away_rest_days"].fillna(7)
    # Elo fill
    feat_df["home_elo"] = feat_df["home_elo"].fillna(1500)
    feat_df["away_elo"] = feat_df["away_elo"].fillna(1500)
    # add home_field indicator (always 1 for home)
    feat_df["is_home"] = 1
    # add feature: elo_diff
    feat_df["elo_diff"] = feat_df["home_elo"] - feat_df["away_elo"]
    # add feature: avg_points_for_diff
    feat_df["pf_diff_last3"] = feat_df["home_avg_pf_last3"] - feat_df["away_avg_pf_last3"]
    return feat_df
