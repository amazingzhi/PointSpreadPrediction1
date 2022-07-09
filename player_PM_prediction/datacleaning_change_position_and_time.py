import csv


def main_process():
    # import data
    PlayerPredictionData = []
    with open(
            '../data/games and player from 2004 to '
            '2020/player_prediction_data/player_prediction_data_after_one_variable_feature_selection.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            PlayerPredictionData.append(row)
    PlayerPredictionDataColumns = PlayerPredictionData[0]
    PlayerPredictionData = PlayerPredictionData[1:]
    # change position and time
    for index, game in enumerate(PlayerPredictionData):
        if game[7] != '0':
            PlayerPredictionData[index][7] = str(1)
        if game[8][1] == ':':
            TempTime = game[8].split(':')
            PlayerPredictionData[index][8] = TempTime[1] + ':' + TempTime[2]
        if game[8].find('60') != -1:
            PlayerPredictionData[index][8].replace('60', '59')
        if len(game[8]) <= 2:
            PlayerPredictionData[index][8] = str(float(game[8]) * 60)
        else:
            obj = game[8].split(':')
            PlayerPredictionData[index][8] = str(float(obj[0]) * 60 + float(obj[1]))
        if game[26][1] == ':':
            TempTime = game[26].split(':')
            PlayerPredictionData[index][26] = TempTime[1] + ':' + TempTime[2]
        if game[26].find('60') != -1:
            PlayerPredictionData[index][26].replace('60', '59')
        if len(game[26]) <= 2:
            PlayerPredictionData[index][26] = str(float(game[26]) * 60)
        else:
            obj = game[26].split(':')
            PlayerPredictionData[index][26] = str(float(obj[0]) * 60 + float(obj[1]))
        if len(game[43]) > 5:
            TempTime = game[43].split(':')
            PlayerPredictionData[index][43] = TempTime[0] + ':' + TempTime[1]
        if len(game[51]) > 5:
            TempTime = game[51].split(':')
            PlayerPredictionData[index][51] = TempTime[0] + ':' + TempTime[1]
        if len(game[58]) > 5:
            TempTime = game[58].split(':')
            PlayerPredictionData[index][58] = TempTime[0] + ':' + TempTime[1]
        if game[43].find('60') != -1:
            PlayerPredictionData[index][43].replace('60', '59')
        if game[51].find('60') != -1:
            PlayerPredictionData[index][51].replace('60', '59')
        if game[58].find('60') != -1:
            PlayerPredictionData[index][58].replace('60', '59')
        if len(game[43]) <= 2:
            PlayerPredictionData[index][43] = str(float(game[43]) * 60)
        else:
            obj = game[43].split(':')
            PlayerPredictionData[index][43] = str(float(obj[0]) * 60 + float(obj[1]))
        if len(game[51]) <= 2:
            PlayerPredictionData[index][51] = str(float(game[51]) * 60)
        else:
            obj = game[51].split(':')
            PlayerPredictionData[index][51] = str(float(obj[0]) * 60 + float(obj[1]))
        if len(game[58]) <= 2:
            PlayerPredictionData[index][58] = str(float(game[58]) * 60)
        else:
            obj = game[58].split(':')
            PlayerPredictionData[index][58] = str(float(obj[0]) * 60 + float(obj[1]))
    PlayerPredictionData.insert(0,PlayerPredictionDataColumns)
    with open('../data/games and player from 2004 to '
              '2020/player_prediction_data/player_prediction_data_feature_selection_and_seconds.csv', mode='w',
              newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        for new_data in PlayerPredictionData:
            csv_writer.writerow(new_data)




if __name__ == "__main__":
    main_process()
