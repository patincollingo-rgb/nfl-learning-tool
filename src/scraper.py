# src/data/scraper.py
import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm

CACHE_DIR = "data_cache"
BASE_URL = "https://www.pro-football-reference.com"

os.makedirs(CACHE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "nfl-learning-tool (+https://github.com/patincollingo-rgb/nfl-learning-tool)"
}

def _cached_path(year):
    return os.path.join(CACHE_DIR, f"games_{year}.html")

def fetch_year_games_page(year, force=False, sleep=1.0):
    """Download and cache the PFR year schedule/games page."""
    path = _cached_path(year)
    if os.path.exists(path) and not force:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    url = f"{BASE_URL}/years/{year}/games.htm"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html = resp.text
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    time.sleep(sleep)
    return html

def parse_games_page(html):
    """Parse PFR year games page into a DataFrame."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="games")
    if table is None:
        raise RuntimeError("Could not find games table on page")
    rows = table.tbody.find_all("tr", recursive=False)
    recs = []
    for r in rows:
        if r.get("class") and "thead" in r.get("class"):
            continue
        cols = [c.get_text(strip=True) for c in r.find_all(["th","td"])]
        # columns change over years; we map by header names below
        recs.append((r, cols))
    # Get headers robustly:
    headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
    # Build DataFrame by parsing each row's named columns using header positions
    data = []
    for r, cols in recs:
        # Create mapping header->cell text (lengths sometimes mismatch; align by index)
        record = {}
        for i, hdr in enumerate(headers):
            if i < len(cols):
                record[hdr] = cols[i]
            else:
                record[hdr] = ""
        # Also extract team links (to get team abbreviations)
        cells = r.find_all(["th","td"])
        # visitor team cell is usually at header "Visitor/tm" or "Visitor"
        # safer: find cells with anchor to /teams/
        team_abbrevs = []
        for c in cells:
            a = c.find("a", href=True)
            if a and "/teams/" in a["href"]:
                # extract first /teams/XXX/ link's XXX
                href = a["href"]
                parts = href.split("/")
                try:
                    idx = parts.index("teams")
                    abb = parts[idx+1]
                    team_abbrevs.append(abb)
                except Exception:
                    pass
        # team_abbrevs often contains visitor then home.
        if len(team_abbrevs) >= 2:
            record["visitor_abbr"], record["home_abbr"] = team_abbrevs[0], team_abbrevs[1]
        else:
            record["visitor_abbr"], record["home_abbr"] = None, None
        data.append(record)
    df = pd.DataFrame(data)
    return df

def get_games_df_for_year(year, force=False):
    html = fetch_year_games_page(year, force=force)
    df = parse_games_page(html)
    # normalize & keep important columns if present
    keep_cols = ["Week","Date","Time","Boxscore","Visitor/tm","Pts","Home/tm","Pts.1","winner/tie","home_abbr","visitor_abbr"]
    # PFR headers may be "Visitor/tm", "Home/tm", "Pts", "Pts.1" etc. Manage gracefully.
    rename_map = {
        "Visitor/tm": "visitor_team",
        "Home/tm": "home_team",
        "Pts": "visitor_points",
        "Pts.1": "home_points",
        "Boxscore": "boxscore",
        "Week": "week",
        "Date": "date",
        "Time": "time"
    }
    df = df.rename(columns=rename_map)
    # convert dates
    if "date" in df.columns:
        def parse_date(s):
            try:
                return pd.to_datetime(s)
            except:
                return pd.NaT
        df["date"] = df["date"].apply(parse_date)
    # numeric points:
    for c in ["visitor_points","home_points"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # week to numeric where possible
    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    df = get_games_df_for_year(args.year, force=False)
    print(df.head())
    df.to_csv(f"data_cache/games_{args.year}.csv", index=False)
