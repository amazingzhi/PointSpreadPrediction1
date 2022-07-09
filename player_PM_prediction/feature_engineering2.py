import csv
import statistics
import numpy as np
OPPO_AVG_COLUMNS = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1]
last_60 = 60
last_30 = 30
last_3 = 3
up_limit = 3


def read_data():
    # import data
    PlayerData = []
    with open('C:/Users/jliu471/Desktop/PlayerDataPreperation.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            PlayerData.append(row)
    PlayerPredictionData = []
    with open('C:/Users/jliu471/Desktop/prediction_keys.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            PlayerPredictionData.append(row)

    Games = []
    with open('H:/python project 1/NBA data/games and player from 2004 to 2020/cleaned_games.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            Games.append(row)
    PlayerPredictionData = PlayerPredictionData[1:]
    print(len(PlayerPredictionData))
    NewData = []
    for index, game in enumerate(PlayerPredictionData):
        if index % 1000 == 0:
            print(index)
        GameID = game[0]
        OppoID = game[3]
        OneObservation = []
        OneObservation = oppoent_average_N_games_sum_and_last_n_games(game_id=GameID, oppo_id=OppoID, N_avg=last_60,
                                                                      Games=Games,
                                                                      PlayerData=PlayerData)[:]
        OneObservation.extend(
            oppoent_average_N_games_sum_and_last_n_games(game_id=GameID, oppo_id=OppoID, N_avg=last_30, Games=Games,
                                                         PlayerData=PlayerData))
        NewData.append(OneObservation)
    with open('player_prediction_data_additional.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for new_data in NewData:
            csv_writer.writerow(new_data)
    ##feed in average last 60 and 30 games sum of features of oppoent
    ## and feed in last three games of oppoent features
    ## (N_avg is 60 or 30 average; N is last N games)


def oppoent_average_N_games_sum_and_last_n_games(game_id, oppo_id, N_avg, Games,
                                                 PlayerData):  # n<=N_avg
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
    if len(OppoLastNGames) != N_avg:
        print(f'gameid {game_id} does not have {N_avg} games before, its length is {len(OppoLastNGames)}.')
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
    return NoPlusMinusAverage


####calculate sum PM by given game id and oppo id
def sum_one_game_PM(game_id, oppo_id, PlayerData):
    # one game's sum PM
    OneGamePMs = []
    for key, gameplayer in enumerate(PlayerData):
        if gameplayer[0] == game_id and gameplayer[1] == oppo_id:
            i = 0
            while PlayerData[key + i][1] == oppo_id:
                OneGamePMs.append(int(PlayerData[key + i][2]))
                i += 1
            break
    if OneGamePMs != []:
        pass
    else:
        print(f'error: can not find gameid: {game_id} with this oppoid: {oppo_id} in PlayerData.')
    SumOneGamePMs = sum(OneGamePMs)
    return SumOneGamePMs
def calculateAverage(aBigList):
    # 82 * 52 list[list[]]
    a = np.array(aBigList)
    a = a.astype(float)
    a = np.mean(a, axis=0)  # axis=0，计算每一列的均值
    a = a.tolist()
    a = list(map(str, a))
    return a

if __name__ == "__main__":
    read_data()
