import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from scipy.stats import norm
from collections import defaultdict
import re

team_map = {
   'ATL': 'Atlanta Hawks',
   'BKN': 'Brooklyn Nets',
   'BOS': 'Boston Celtics', 
   'CHA': 'Charlotte Hornets',
   'CHI': 'Chicago Bulls',
   'CLE': 'Cleveland Cavaliers',
   'DAL': 'Dallas Mavericks',
   'DEN': 'Denver Nuggets',
   'DET': 'Detroit Pistons',
   'GSW': 'Golden State Warriors',
   'HOU': 'Houston Rockets',
   'IND': 'Indiana Pacers',
   'LAC': 'Los Angeles Clippers',
   'LAL': 'Los Angeles Lakers',
   'MEM': 'Memphis Grizzlies',
   'MIA': 'Miami Heat',
   'MIL': 'Milwaukee Bucks',
   'MIN': 'Minnesota Timberwolves',
   'NOP': 'New Orleans Pelicans',
   'NYK': 'New York Knicks',
   'ORL': 'Orlando Magic',
   'PHI': 'Philadelphia 76ers',
   'PHX': 'Phoenix Suns',
   'POR': 'Portland Trail Blazers',
   'SAC': 'Sacramento Kings',
   'SAS': 'San Antonio Spurs',
   'TOR': 'Toronto Raptors',
   'UTA': 'Utah Jazz',
   'WAS': 'Washington Wizards',
   'OKC': 'Oklahoma City Thunder'
}

import re
from collections import defaultdict

# Input data as a multiline string
data = """
OKC
OKC
POR
POR
Expected Lineup
PG S. Gilgeous-Alexander
SG Cason Wallace
SF Aaron Wiggins
PF J. Williams
C I. Hartenstein
MAY NOT PLAY
F L. Dort OUT
C C. Holmgren OUT
G A. Mitchell OUT
Expected Lineup
PG A. Simons
SG T. Camara
SF Deni Avdija
PF Jerami Grant
C D. Clingan
MAY NOT PLAY
C D. Ayton OUT
F M. Thybulle OUT
"""

def parse_lineup_status():
   """Parse player lineup data and return players listed as OUT"""
   lines = data.split("\n")
   out_players = []
   
   for line in lines:
       line = line.strip()
       if not line:
           continue
           
       elif re.match(r"(PG|SG|SF|PF|C|G|F)", line):
           if len(line.split()) > 3 and line.split()[3] == 'OUT':
               out_players.append(f'{line.split()[1][:1]}{line.split()[2]}')
               
   return out_players

def get_home_team(team1, team2, data):
    """
    Determines the home team based on which team appears first in the data string.
    """
    team1_index = data.find(team1)
    team2_index = data.find(team2)

    if team1_index == -1 or team2_index == -1:
        raise ValueError("One or both teams not found in data")

    return team1 if team1_index > team2_index else team2

def compute_dynamic_elo_home_away(
    games_df: pd.DataFrame,
    date_col: str = 'Date',
    home_team_col: str = 'Home',
    away_team_col: str = 'Visitor',
    home_score_col: str = 'HomePTS',
    away_score_col: str = 'VisitorPTS',
    initial_elo: float = 1500.0,
    K_min: float = 10.0,
    K_max: float = 30.0
):
    games_df[date_col] = pd.to_datetime(games_df[date_col])
    games_df = games_df.sort_values(by=date_col).reset_index(drop=True)

    all_teams = pd.concat([
        games_df[home_team_col],
        games_df[away_team_col]
    ]).unique()

    team_elos = {
        team: {
            'home': initial_elo,
            'away': initial_elo
        }
        for team in all_teams
    }

    total_games = len(games_df)

    def expected_score(rating_a, rating_b):
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def get_dynamic_k_factor(game_index, total_count, k_min, k_max):
        if total_count <= 1:
            return k_min  
        progress = game_index / (total_count - 1)
        return k_min + progress * (k_max - k_min)

    history_records = []

    for idx, row in games_df.iterrows():
        home_team = row[home_team_col]
        away_team = row[away_team_col]
        home_score = row[home_score_col]
        away_score = row[away_score_col]

        home_elo_before = team_elos[home_team]['home']
        away_elo_before = team_elos[away_team]['away']

        current_k = get_dynamic_k_factor(idx, total_games, K_min, K_max)

        home_expected = expected_score(home_elo_before, away_elo_before)
        away_expected = 1.0 - home_expected

        if home_score > away_score:
            home_actual = 1.0
            away_actual = 0.0
        else:
            home_actual = 0.0
            away_actual = 1.0

        home_elo_after = home_elo_before + current_k * (home_actual - home_expected)
        away_elo_after = away_elo_before + current_k * (away_actual - away_expected)

        team_elos[home_team]['home'] = home_elo_after
        team_elos[away_team]['away'] = away_elo_after

        history_records.append({
            date_col: row[date_col],
            home_team_col: home_team,
            away_team_col: away_team,
            'Home Elo Before': home_elo_before,
            'Away Elo Before': away_elo_before,
            home_score_col: home_score,
            away_score_col: away_score,
            'Home Elo After': home_elo_after,
            'Away Elo After': away_elo_after,
            'K Factor Used': current_k
        })

    elo_history_df = pd.DataFrame(history_records)

    return elo_history_df, team_elos

# NAME,Team,OPP,Pos,MIN,PTS,REB,AST,STL,BLK,TO,FGM,FGA,FG%,3PM,3PA,3P%,FTM,FTA,FT%,OREB,DREB
def simulate_games(game, out_list, final_elos):
    game['scratch'] = game.apply(lambda x: f"{x['NAME'].split()[0][0]}{x['NAME'].split()[1]}" not in out_list, axis=1)
    game = game[game['scratch'] == True]

    game = game.copy()
    game['full_team'] = game['Team'].map(team_map)
    game['full_opp'] = game['OPP'].map(team_map)
    
    game['hometeam'] = game.apply(lambda x: x['Team'] == get_home_team(x['Team'], x['OPP'], data), axis=1)

    def calculate_elo_advantage(row):
        home_advantage = final_elos[row['full_team']]['home'] if row['hometeam'] else final_elos[row['full_team']]['away']
        away_disadvantage = final_elos[row['full_opp']]['home'] if row['hometeam'] else final_elos[row['full_opp']]['away']
        return home_advantage - away_disadvantage

    game['elo_advantage'] = game.apply(calculate_elo_advantage, axis=1)

    def adjust_points(row):
        base_pts = row['PTS']
        elo_adjustment = row['elo_advantage'] * 0.02
        home_bonus = 3 if row['hometeam'] else 0
        return base_pts + elo_adjustment + home_bonus

    game['adjusted_PTS'] = game.apply(adjust_points, axis=1)

    matchups = game[['full_team', 'full_opp']].drop_duplicates().values
    unique_games = []

    for team1, team2 in matchups:
        if f"{team2} vs {team1}" not in unique_games:
            unique_games.append(f"{team1} vs {team2}")

    for g in unique_games:
        teams = g.split(' vs ')
        print(f"\nScores for {teams[0]} and {teams[1]}:")

        # Store the scores in a dictionary, so we can compare them later
        scores = {}
        for team in teams:
            team_score = game[game['full_team'] == team]['adjusted_PTS'].sum()
            scores[team] = round(team_score, 2)
            print(f"{team}: {scores[team]}")
def main():

    games_directory = r'D:\SportsBetting\2025\basketball\roto\rotowire-nba-projections.csv'
    out_list = parse_lineup_status()
    _, final_elos = compute_dynamic_elo_home_away(games_df=pd.read_csv(r'D:\SportsBetting\2025\basketball\roto\season.csv'))

    game = pd.read_csv(games_directory)
    simulate_games(game, out_list, final_elos)

if __name__ == "__main__":
    main()