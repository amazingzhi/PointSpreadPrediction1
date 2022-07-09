import example_web as ew
import pandas as pd
import os
import glob
import csv
import matplotlib.pyplot as plt

prediction_path = 'predictions_need_accuracies'

all_files = glob.glob(os.path.join(prediction_path, "*.csv"))
data_dic = {}
for file in all_files:
    with open(file) as file_object:
        contents = file_object.read()
        data = contents.split('\n')
        temp = file.split('\\')[-1]
        column = temp.split('.')[0]
        data_dic[column] = data
df = pd.DataFrame.from_dict(data_dic)
df = df.drop([0])
df = df.drop([1333])

df_true = pd.read_csv('../data/games and player from 2004 to 2020/game_player_prediction_data/pca_train.csv')
df_true = df_true[df_true['GAME_DATE_EST'].str.contains('2019')]
y_true = df_true['pointspread']

for column in df:
    print(column)
    ew.calculate_metrics_true_pred(y_true, df[column])