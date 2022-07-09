import pandas as pd
from collections import defaultdict

# import data
betting_data = pd.read_csv('data/betting spread/betting_line_2019.csv', dtype=str)
games_data = pd.read_csv('data/games and player from 2004 to 2020/original_data/games.csv', dtype=str)
games_data = games_data.dropna()
games_data = games_data[games_data['GAME_DATE_EST'].str.contains('2018|2019|2020')]
mapping_data = pd.read_csv('data/games and player from 2004 to 2020/original_data/teams.csv', dtype=str)

# add game ID
mapping = {
    'ATL': 'atl', 'BOS': 'bos', 'NOP': 'no', 'CHI': 'chi', 'DAL': 'dal', 'DEN': 'den', 'HOU': 'hou', 'LAC': 'lac',
    'LAL': 'lal',
    'MIA': 'mia', 'MIL': 'mil', 'MIN': 'min', 'BKN': 'bkn', 'NYK': 'ny', 'ORL': 'orl', 'IND': 'ind', 'PHI': 'phi',
    'PHX': 'phx',
    'POR': 'por', 'SAC': 'sac', 'SAS': 'sa', 'OKC': 'okc', 'TOR': 'tor', 'UTA': 'utah', 'MEM': 'mem', 'WAS': 'wsh',
    'DET': 'det',
    'CHA': 'cha', 'CLE': 'cle', 'GSW': 'gs'
}

mapping_TeamID_TeamName = defaultdict()
for ind_team in mapping_data.index:
    mapping_TeamID_TeamName[mapping_data['ABBREVIATION'][ind_team]] = mapping_data['TEAM_ID'][ind_team]

homeids = []
awayids = []
game_ids = []
for ind_betting in betting_data.index:
    away = list(mapping.keys())[list(mapping.values()).index(betting_data['away'][ind_betting])]
    home = list(mapping.keys())[list(mapping.values()).index(betting_data['home'][ind_betting])]
    away_id = mapping_TeamID_TeamName[away]
    home_id = mapping_TeamID_TeamName[home]
    homeids.append(home)
    awayids.append(away)
    away_point = str(betting_data['result'][ind_betting]).split('-')[0]
    home_point = str(betting_data['result'][ind_betting]).split('-')[1]
    i = 0
    for ind_games in games_data.index:
        if home_id == games_data['HOME_TEAM_ID'][ind_games] and away_id == games_data['VISITOR_TEAM_ID'][ind_games] and home_point == games_data['PTS_home'][ind_games] and away_point == games_data['PTS_away'][ind_games]:
            game_ids.append(games_data['GAME_ID'][ind_games])
            i += 1
    if i > 1:
        print(f'error: {home_id} vs {away_id}: {home_point}-{away_point} has more than one game id.')
        print(f'index: from {game_ids[-i]} to {game_ids[-1]}')
    if i == 0:
        print(f'cannot find {home_id} vs {away_id}: {home_point}-{away_point}')
        print(f'index from {game_ids[-1]}')

betting_data['home_id'] = homeids
betting_data['away_id'] = awayids
resultCSVPath = 'data/betting spread/cleaned_betting_line_2019.csv'
betting_data.to_csv(resultCSVPath,index = False,na_rep = 0)
resultCSVPath = 'data/betting spread/game_ids.csv'
game_ids.to_csv(resultCSVPath,index = False,na_rep = 0)


