# src/utils/elo.py
import math

DEFAULT_ELO = 1500

class Elo:
    def __init__(self, k=20, home_field=20, initial=DEFAULT_ELO):
        self.k = k
        self.home_field = home_field
        self.ratings = {}
        self.initial = initial

    def _get(self, team):
        return self.ratings.get(team, self.initial)

    def expected(self, ra, rb):
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        return ea

    def update(self, home_team, away_team, home_score, away_score):
        ra = self._get(home_team) + self.home_field
        rb = self._get(away_team)
        ea = self.expected(ra, rb)
        # determine actual score for home team
        if home_score > away_score:
            sa = 1.0
        elif home_score == away_score:
            sa = 0.5
        else:
            sa = 0.0
        # Update base ratings (remove home_field before storing)
        adj_home = self._get(home_team)
        adj_away = self._get(away_team)
        new_home = adj_home + self.k * (sa - ea)
        # update away symmetrical
        # expected away is 1 - ea
        new_away = adj_away + self.k * ((1 - sa) - (1 - ea))
        self.ratings[home_team] = new_home
        self.ratings[away_team] = new_away

    def get(self, team):
        return self._get(team)
