# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 14:45:44 2021

@author: jliu471
"""
# import libraries
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statistics

# global variables
last_60 = 60
last_30 = 30
last_3 = 3
up_limit = 3
##column names
###columns names set up
Original_COLUMNS = ['GAME_DATE_EST', 'GAME_ID', 'TEAM_ID', 'OPPOID', 'TEAM_ABBREVIATION', 'TEAM_CITY', 'PLAYER_ID',
                    'PLAYER_NAME', 'START_POSITION', 'COMMENT', 'H_or_A', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                    'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS',
                    'PLUS_MINUS', 'SEASON_ID']
Original_Games_Columns = ['GAME_DATE_EST', 'GAME_ID', 'GAME_STATUS_TEXT', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID',
                          'SEASON', 'TEAM_ID_home', 'PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home',
                          'AST_home', 'REB_home', 'TEAM_ID_away', 'PTS_away', 'FG_PCT_away', 'FT_PCT_away',
                          'FG3_PCT_away', 'AST_away', 'REB_away', 'HOME_TEAM_WINS', 'SEASON_TYPE', 'pointspread'
                          ]
###avg 60 or 30 or sum average oppoent data columns
AVG_60_or_30_COLUMNS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        0]

###last 3 games columns or last 3 games sum oppoent data columns
LAST_3_GAMES_COLUMNS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 0]
###oppoent average games features
OPPO_AVG_COLUMNS = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1]

####average 60's columns

TempCOLMNS = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
              'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS',
              'PLUS_MINUS']


def read_data():
    # import data
    PlayerData = []
    with open('../data/games and player from 2004 to 2020/completed_cleaned_player_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            PlayerData.append(row)
    Players = []
    with open('../data/games and player from 2004 to 2020/players.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            Players.append(row)
    Games = []
    with open('../data/games and player from 2004 to 2020/games.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            Games.append(row)

    # #minor change of original player-level data
    PlayerData = PlayerData[1:]
    Games = Games[1:]


    # build a player dictionary to allocate all players
    ##find all players
    PlayerDic = {}
    for player in Players:
        PlayerDic[player[2]] = []
    ### drop colunm name of this dictionary
    del PlayerDic['PLAYER_ID']

    ##find all games each player played
    PlayerList = []
    for player in PlayerDic.keys():
        PlayerList.append(player)
    for player in PlayerList:
        for game in PlayerData:
            if player == game[6]:
                PlayerDic[player].append(game)

    ## drop players that don't play 65 games
    PlayersToBeRemove = []
    for player, games in PlayerDic.items():
        if len(games) < 65:
            PlayersToBeRemove.append(player)
    for player in PlayersToBeRemove:
        del PlayerDic[player]

    # build a new list to build a new dataset to make prediction
    PlayerPredictionData = []
    for player, games in PlayerDic.items():
        for index, game in enumerate(games):
            if len(games) - 1 - int(index) >= 60:
                PlayerPredictionData.append([game[1], game[6], game[2], game[3], game[10], game[0], game[8], game[30]])
    print(len(PlayerPredictionData))
    # colunm names generation

    AVG_60_COLUMNS = []
    for i in TempCOLMNS:
        AVG_60_COLUMNS.append('Avg60' + i)
    ####average 30's colunms
    AVG_30_COLUMNS = []
    for i in TempCOLMNS:
        AVG_30_COLUMNS.append('Avg30' + i)
    ####player last three games columns
    player_last_three_games_colunms = []
    TempCOLMNS1 = []
    for index, flag in enumerate(LAST_3_GAMES_COLUMNS):
        if flag == 1:
            TempCOLMNS1.append(Original_COLUMNS[index])
    for index, flag in enumerate(LAST_3_GAMES_COLUMNS[12:]):
        if flag == 1:
            TempCOLMNS1.append('intercept' + Original_COLUMNS[index])
    for i in range(3):
        for name in TempCOLMNS1:
            player_last_three_games_colunms.append('Lag' + str(i + 1) + name)
    ####oppoent's last 60 and 30 average colunms and oppoent's last three games colunms
    TempCOLMNS2 = ['PLUS_MINUS']
    for index, flag in enumerate(OPPO_AVG_COLUMNS):
        if flag == 1:
            TempCOLMNS2.append(Original_Games_Columns[index])
    oppo_avg_60_colunms = []
    oppo_avg_30_colunms = []
    for i in TempCOLMNS2:
        oppo_avg_60_colunms.append('OppoAvg60' + i)
        oppo_avg_30_colunms.append('OppoAvg30' + i)
    TempCOLMNS3 = ['PLUS_MINUS', 'Loc']
    for index, flag in enumerate(OPPO_AVG_COLUMNS):
        if flag == 1:
            TempCOLMNS3.append(Original_Games_Columns[index])
    oppo_last_3_colunms = []
    for i in range(3):
        for name in TempCOLMNS3:
            oppo_last_3_colunms.append('OppoLag' + str(i + 1) + name)
    ####player vs oppoent's last 2 games colunms
    avg_player_vs_oppo_colunms = []
    for name in TempCOLMNS[1:]:
        avg_player_vs_oppo_colunms.append('AvgPlayerVSOppo' + name)
    ###merge these colunms
    Merged_Colunms = ['GAME_ID', 'PLAYER_ID', 'TEAM_ID', 'OPPOID', 'LOC', 'GAME_DATE_EST', 'START_POSITION',
                      'PLUS_MINUS']
    Merged_Colunms.extend(AVG_60_COLUMNS)
    Merged_Colunms.extend(AVG_30_COLUMNS)
    Merged_Colunms.extend(player_last_three_games_colunms)
    Merged_Colunms.extend(oppo_avg_60_colunms)
    Merged_Colunms.extend(oppo_avg_30_colunms)
    Merged_Colunms.extend(oppo_last_3_colunms)
    Merged_Colunms.extend(avg_player_vs_oppo_colunms)

    ##merge player data main

    NewData = []
    NewData.append(Merged_Colunms)

    for index, game in enumerate(PlayerPredictionData):
        GameId = game[0]
        PlayerId = game[1]
        OneObservation = []
        if last_n_games_player_vs_oppo(game_id=GameId, player_id=PlayerId, n=up_limit, PlayerDic=PlayerDic) == False:
            print(index)
            continue
        else:
            OneObservation = game[:]
            OneObservation.extend(player_average(game_id=GameId, player_id=PlayerId, n=last_60, PlayerDic=PlayerDic))
            OneObservation.extend(player_average(game_id=GameId, player_id=PlayerId, n=last_30, PlayerDic=PlayerDic))
            OneObservation.extend(player_last_three_games(game_id=GameId, player_id=PlayerId, n=last_3, PlayerDic=PlayerDic))
            # OneObservation.extend(
            #     oppoent_average_N_games_sum_and_last_n_games(game_id=GameId, player_id=PlayerId, N_avg=last_60,
            #                                                  n=last_3, PlayerDic=PlayerDic, Games=Games,PlayerData=PlayerData)[0])
            # OneObservation.extend(
            #     oppoent_average_N_games_sum_and_last_n_games(game_id=GameId, player_id=PlayerId, N_avg=last_30,
            #                                                  n=last_3, PlayerDic=PlayerDic, Games=Games,PlayerData=PlayerData)[0])
            # OneObservation.extend(
            #     oppoent_average_N_games_sum_and_last_n_games(game_id=GameId, player_id=PlayerId, N_avg=last_30,
            #                                                  n=last_3, PlayerDic=PlayerDic, Games=Games,PlayerData=PlayerData)[1])
            OneObservation.extend(last_n_games_player_vs_oppo(game_id=GameId, player_id=PlayerId, n=up_limit, PlayerDic=PlayerDic))
            NewData.append(OneObservation)
    with open('../data/player/player_prediction_data.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        for new_data in NewData:
            csv_writer.writerow(new_data)

##feed in averge last 60 and 30 games data for that player of that game 
def player_average(game_id, player_id, n, PlayerDic):  # (n<=60)
    AverageList = []
    PastNGamesNoMins = []
    Min = []
    for player, games in PlayerDic.items():
        if player == player_id:
            for index, game in enumerate(games):
                if game[1] == game_id:
                    for i in range(n):
                        # find player_MIN and convert to datetime and calculate average
                        # find last n games minutes' string
                        Min.append(games[index + 1 + i][11])
                        # find other player features and calculate average

                        OneGameNoMin = []
                        for index1, flag in enumerate(AVG_60_or_30_COLUMNS):
                            if flag == 1:
                                OneGameNoMin.append(games[index + 1 + i][index1])
                        PastNGamesNoMins.append(OneGameNoMin)
    if len(PastNGamesNoMins) < n:
        print('error: games is less than ' + n)
    ##calculate no minutes average data
    AverageList = calculateAverage(PastNGamesNoMins)
    ##delete milliseconds of this string
    NoMilliseceondsString = delete_milliseconds_of_strings(Min)
    ##convert string to timedelta
    ListOfTimedeltas = convert_list_of_strings_to_time(NoMilliseceondsString)
    ##calculate average timedelta
    AveragedTimedelta = average_timedelta(ListOfTimedeltas)
    # merge min and no min together
    AverageList.insert(0, str(AveragedTimedelta))
    return AverageList


##计算每一列的均值
def calculateAverage(aBigList):
    # 82 * 52 list[list[]]
    a = np.array(aBigList)
    a = a.astype(float)
    a = np.mean(a, axis=0)  # axis=0，计算每一列的均值
    a = a.tolist()
    a = list(map(str, a))
    return a


## calculate average time spent for each player's games rolling
def average_timedelta(list_of_timedeltas):
    AverageTimedelta = sum(list_of_timedeltas, timedelta(0)) / len(list_of_timedeltas)
    return AverageTimedelta


def convert_list_of_strings_to_time(strings):
    list_of_timedeltas = []
    for string in strings:
        if len(string) <= 2:
            obj = datetime.strptime(string, '%M')
            delta = timedelta(minutes=obj.minute)
        else:
            obj = datetime.strptime(string, '%M:%S')
            delta = timedelta(minutes=obj.minute, seconds=obj.second)
        list_of_timedeltas.append(delta)
    return list_of_timedeltas


def delete_milliseconds_of_strings(strings):
    new_strings = []
    for string in strings:
        if len(string) > 5:
            TempTime = string.split(':')
            new_strings.append(TempTime[0] + ':' + TempTime[1])
        else:
            new_strings.append(string)
    new_strings_one = []
    for string in new_strings:
        if string.find('60') == -1:
            new_strings_one.append(string)
        else:
            new_strings_one.append(string.replace('60', '59'))
    return new_strings_one


##feed in last n games of that player before that game
def player_last_three_games(game_id, player_id, n, PlayerDic):
    LastThreeGames = []
    for player, games in PlayerDic.items():
        if player == player_id:
            for index, game in enumerate(games):
                if game[1] == game_id:
                    for i in range(n):
                        OneGame = []
                        for index1, flag in enumerate(LAST_3_GAMES_COLUMNS):
                            if flag == 1:
                                OneGame.append(games[index + 1 + i][index1])
                        for index1, flag in enumerate(LAST_3_GAMES_COLUMNS[12:]):
                            if flag == 1:
                                OneGame.append(str(float(games[index + 1 + i][index1+12]) * float(games[index + 1 + i][10])))
                        LastThreeGames.extend(OneGame)
    return LastThreeGames


##feed in average last 60 and 30 games sum of features of oppoent
## and feed in last three games of oppoent features
## (N_avg is 60 or 30 average; N is last N games)
def oppoent_average_N_games_sum_and_last_n_games(game_id, player_id, N_avg, n, PlayerDic, Games, PlayerData):  # n<=N_avg
    ###get oppo id for one game and one player
    for player, games in PlayerDic.items():
        if player == player_id:
            for index, game in enumerate(games):
                if game[1] == game_id:
                    oppo_id = game[3]
    ###get last N_avg games this oppoent played
    OppoLastNGames = []
    GameIDs = []
    for j, line in enumerate(Games):
        Game_ID = line[1]
        if Game_ID == game_id:
            for i in range(j + 1, len(Games) - 1):
                if Games[i][3] == oppo_id or Games[i][4] == oppo_id:
                    GameIDs.append(Games[i][1])
                    OppoLastNGames.append(Games[i])
                if len(OppoLastNGames) >= N_avg:
                    break

    ###get required columns' averaged data
    OppoLastNGamesRequiredColumns = []
    for index, game in enumerate(OppoLastNGames):
        if game[3] == oppo_id:
            OneHomeGame = []
            for index1, flag in enumerate(OPPO_AVG_COLUMNS):
                if flag == 1:
                    OneHomeGame.append(game[index1])
            OppoLastNGamesRequiredColumns.append(OneHomeGame)
        elif game[4] == oppo_id:
            OppoLastNGamesRequiredColumns.append([game[14], game[15], game[16], game[17], game[18], game[19],
                                                  game[7], game[8], game[9], game[10], game[11], game[12],
                                                  str(-int(game[22]))])
        else:
            print('error')
    NoPlusMinusAverage = calculateAverage(OppoLastNGamesRequiredColumns)
    ### get one extra average plus minus and then merge back to NoPlusMinusAverage
    NGamesPM = []
    for GameID in GameIDs:
        NGamesPM.append(sum_one_game_PM(game_id=GameID, oppo_id=oppo_id, PlayerData=PlayerData))
    AvgNGamePM = str(statistics.mean(NGamesPM))
    ###insert PM into other average data
    NoPlusMinusAverage.insert(0, AvgNGamePM)

    ###get required column's last n games data
    ####find oppoent last n games' whole line
    OppoLastnGames = []
    for i in range(n):
        OppoLastnGames.append(OppoLastNGames[i])
    ####find the required columns from the whole line
    OppoLastnGamesRequiredColumns = []
    for index, game in enumerate(OppoLastnGames):
        # if oppoent id is the home team of that line, get required colunms based on above. str(1) shows it is home team
        if game[3] == oppo_id:
            OneHomeGame = [str(1)]
            for index1, flag in enumerate(OPPO_AVG_COLUMNS):
                if flag == 1:
                    OneHomeGame.append(game[index1])
            OppoLastnGamesRequiredColumns.append(OneHomeGame)
        # if oppoent id is the away team of that line, set up reverse colunms by hands. str(0) shows it is away team
        elif game[4] == oppo_id:
            OppoLastnGamesRequiredColumns.append([str(0), game[14], game[15], game[16], game[17], game[18], game[19],
                                                  game[7], game[8], game[9], game[10], game[11], game[12],
                                                  str(-int(game[22]))])
        # for debug
        else:
            print('error')
    ####get last n games' sum plus minus and insert to required colunms
    OppoLastnGamesIDs = []
    for i in range(n):
        OppoLastnGamesIDs.append(GameIDs[i])
    nGamesPMs = []
    for GameID in OppoLastnGamesIDs:
        nGamesPMs.append(sum_one_game_PM(game_id=GameID, oppo_id=oppo_id,PlayerData=PlayerData))
    for i in range(n):
        OppoLastnGamesRequiredColumns[i].insert(0, nGamesPMs[i])
    return NoPlusMinusAverage, OppoLastnGamesRequiredColumns


####calculate sum PM by given game id and oppo id
def sum_one_game_PM(game_id, oppo_id, PlayerData):
    # one game's sum PM
    OneGamePMs = []
    for key, gameplayer in enumerate(PlayerData):
        if gameplayer[1] == game_id and gameplayer[2] == oppo_id:
            for i in range(20):
                if PlayerData[key + i][2] == oppo_id:
                    OneGamePMs.append(int(PlayerData[key + i][-2]))
                else:
                    break
            break
    SumOneGamePMs = sum(OneGamePMs)
    return SumOneGamePMs


##feed in past n games of this player vs specific team
def last_n_games_player_vs_oppo(game_id, player_id, n,PlayerDic):  # n is the up limit games
    LastALLGamesvsOppo = []
    for player, games in PlayerDic.items():
        if player == player_id:
            for key, game in enumerate(games):
                if game[1] == game_id:
                    for i in range(key + 1, len(games) - 1):
                        if games[i][3] == game[3]:
                            LastALLGamesvsOppo.append(games[i])
    LastNGamesvsOppo = []
    AvgNGamesOppo = []
    if len(LastALLGamesvsOppo) > 1 and len(LastALLGamesvsOppo) <= n:
        for i in range(len(LastALLGamesvsOppo)):
            LastOneGamesvsOppo = []
            for index, flag in enumerate(AVG_60_or_30_COLUMNS):
                if flag == 1:
                    LastOneGamesvsOppo.append(LastALLGamesvsOppo[i][index])
            LastNGamesvsOppo.append(LastOneGamesvsOppo)
        AvgNGamesOppo = calculateAverage(LastNGamesvsOppo)
    elif len(LastALLGamesvsOppo) > n:
        for i in range(n):
            LastOneGamesvsOppo = []
            for index, flag in enumerate(AVG_60_or_30_COLUMNS):
                if flag == 1:
                    LastOneGamesvsOppo.append(LastALLGamesvsOppo[i][index])
            LastNGamesvsOppo.append(LastOneGamesvsOppo)
        AvgNGamesOppo = calculateAverage(LastNGamesvsOppo)
    elif len(LastALLGamesvsOppo) <= 1 and len(LastALLGamesvsOppo) > 0:
        for index, flag in enumerate(AVG_60_or_30_COLUMNS):
            if flag == 1:
                AvgNGamesOppo.append(LastALLGamesvsOppo[0][index])
    else:
        print('no game vs this oppo')
        return False
    return AvgNGamesOppo


if __name__ == "__main__":
    read_data()
