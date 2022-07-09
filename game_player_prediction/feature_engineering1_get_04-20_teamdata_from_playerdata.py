import csv

import numpy as np


def read_data():
    PlayerData = []
    with open('../data/games and player from 2004 to '
              '2020/completed_cleaned_player_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            PlayerData.append(row)
    PlayerData = PlayerData[1:]
    Games = []
    with open('../data/games and player from 2004 to '
              '2020/game_prediction_data/cleaned_games.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            Games.append(row)
    Games = Games[1:]
    def generate_one_observation(gameid,homeid,awayid):

        OneObservation_H = []
        OneObservation_A = []
        for index, value in enumerate(PlayerData):
            if value[1] == gameid:
                if value[2] == homeid:
                    OneLine_H = value[12:14]
                    OneLine_H.extend(value[15:17])
                    OneLine_H.extend(value[18:20])
                    OneLine_H.extend(value[21:23])
                    OneLine_H.extend(value[25:])
                    OneObservation_H.append(OneLine_H)
                elif value[2] == awayid:
                    OneLine_A = value[12:14]
                    OneLine_A.extend(value[15:17])
                    OneLine_A.extend(value[18:20])
                    OneLine_A.extend(value[21:23])
                    OneLine_A.extend(value[25:])
                    OneObservation_A.append(OneLine_A)
                else:
                    print(f'error, value[1] is a problem')
        OneObservation_H = sum(OneObservation_H)
        OneObservation_A = sum(OneObservation_A)
        OneObservation_H.extend(OneObservation_A)
        return OneObservation_H
    NewData = []
    for i,v in enumerate(Games):
        gameid = v[1]
        homeid = v[3]
        awayid = v[4]
        OneObservation = generate_one_observation(gameid,homeid,awayid)
        v.extend(OneObservation)
        NewData.append(v)
    with open('../data/games and player from 2004 to '
              '2020/game_prediction_data/complete_0420_team_data.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        for new_data in NewData:
            csv_writer.writerow(new_data)



def sum(Alist):
    ndarray = np.array(Alist)
    ndarray = ndarray.astype(float)
    sumed_np = np.sum(ndarray, axis=0)
    sumed_np = sumed_np.tolist()
    sumed_np = list(map(str, sumed_np))
    return sumed_np
if __name__ == "__main__":
    read_data()
