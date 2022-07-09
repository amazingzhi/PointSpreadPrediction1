import copy
import csv

def calculate_one_team_PM(gameid,homeid,awayid,Data):
    HomePMs= []
    AwayPMs = []
    for i,game in enumerate(Data):
        if game[0] == gameid:
            if game[2] == homeid:
                HomePMs.append(float(game[-1]))
            elif game[2] == awayid:
                AwayPMs.append((float(game[-1])))
            else:
                print(f'error: this {gameid} is wrong!!!')
    Dif_H_A_PM = str(sum(HomePMs) - sum(AwayPMs))
    SumHomePM = str(sum(HomePMs))
    SumAwayPM = str(sum(AwayPMs))
    SumPM = [SumHomePM,SumAwayPM,Dif_H_A_PM]
    return SumPM

def main_process():
# import data
    PlayerPredictedData = []
    with open('../data/games and player from 2004 to '
              '2020/player_prediction_data/player_prediction_data_feature_selection_and_seconds.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            PlayerPredictedData.append(row)
    PlayerPredictedData = PlayerPredictedData[1:]

    player_predicted = []
    with open('../player_PM_prediction/predictions/original_DNN_standard.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            player_predicted.append(row)
    player_predicted = player_predicted[1:]

    for index,line in enumerate(PlayerPredictedData):
        line.append(player_predicted[index][0])

    Games = []
    with open('../data/games and player from 2004 to '
              '2020/game_prediction_data/0420_with_back82_back41_back3_back2.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            Games.append(row)
    for index, game in enumerate(Games):
        if index == 0:
            game.extend(['Sum_H_PM', 'Sum_A_PM', 'Dif_HA_PM'])
        else:
            game_copy = game.copy()
            GameID = game_copy[1]
            HomeID = game_copy[2]
            AwayID = game_copy[3]
            game.extend(calculate_one_team_PM(GameID,HomeID,AwayID,PlayerPredictedData))
    with open('../data/games and player from 2004 to '
              '2020/game_prediction_data/0420_with_back82_back41_back3_back2_SumPMs.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        for new_data in Games:
            csv_writer.writerow(new_data)

if __name__ == "__main__":
    main_process()
