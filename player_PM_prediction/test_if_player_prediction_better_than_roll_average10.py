# import csv
# from statistics import mean
#
# import torch
# def main_process():
#     # import data
#     PlayerData = []
#     with open('H:/python project 1/NBA data/games and player from 2004 to '
#               '2020/player_prediction_data_feature_selection_and_seconds.csv') as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         for row in csv_reader:
#             PlayerData.append(row)
#     PlayerData = PlayerData[1:]
#     PMs = []
#     Avg9PMs = []
#     Avg6PMs = []
#     Avg3PMs = []
#     for index,game in enumerate(PlayerData[:-10]):
#         PMs.append(int(game[5]))
#         Avg9PM = []
#         Avg6PM = []
#         Avg3PM = []
#         for i in range(1,10):
#             if PlayerData[index+i][1] == PlayerData[index][1]:
#                 Avg9PM.append(int(PlayerData[index + i][5]))
#             else:
#                 if not Avg9PM:
#                     Avg9PM = [0]
#                 break
#         Avg9PMs.append(mean(Avg9PM))
#         for i in range(1, 7):
#             if PlayerData[index+i][1] == PlayerData[index][1]:
#                 Avg6PM.append(int(PlayerData[index + i][5]))
#             else:
#                 if not Avg6PM:
#                     Avg6PM = [0]
#                 break
#         Avg6PMs.append(mean(Avg6PM))
#         for i in range(1, 4):
#             if PlayerData[index+i][1] == PlayerData[index][1]:
#                 Avg3PM.append(int(PlayerData[index + i][5]))
#             else:
#                 if not Avg3PM:
#                     Avg3PM = [0]
#                 break
#         Avg3PMs.append(mean(Avg3PM))
#     # loaded_model = torch.load('model1.pth')
#     # loaded_model.eval()
#     # y_hat = loaded_model()
#     result_metrics_9 = calculate_metrics(PMs, Avg9PMs)
#     result_metrics_6 = calculate_metrics(PMs, Avg6PMs)
#     result_metrics_3 = calculate_metrics(PMs, Avg3PMs)
#     result_metrics_9
#     result_metrics_6
#     result_metrics_3

# if __name__ == "__main__":
#     main_process()

import pandas as pd
import example_web as ew
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error
import torch.nn as nn
## see if cuda available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")
class DNNModel(nn.Module):
    def __init__(self, input_size_avg, hidden_size_avg, input_size_lag, hidden_size_lag,
                 input_size_vs_oppo, hidden_size_vs_oppo, input_oppo, hidden_oppo, current_size, hidden_size1,
                 hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6, hidden_size7,
                 hidden_size8, hidden_size9, out_size):
        super(DNNModel, self).__init__()
        self.input_size_avg = input_size_avg
        self.hidden_size_avg = hidden_size_avg
        self.input_size_lag = input_size_lag
        self.hidden_size_lag = hidden_size_lag
        self.input_size_vs_oppo = input_size_vs_oppo
        self.hidden_size_vs_oppo = hidden_size_vs_oppo
        self.input_oppo = input_oppo
        self.hidden_oppo = hidden_oppo
        self.current_size = current_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.out_size = out_size
        self.avg = nn.Linear(input_size_avg, hidden_size_avg)
        self.lag = nn.Linear(input_size_lag, hidden_size_lag)
        self.vs_oppo = nn.Linear(input_size_vs_oppo, hidden_size_vs_oppo)
        self.oppo = nn.Linear(input_oppo, hidden_oppo)
        self.leaky_relu = nn.LeakyReLU()
        self.l1 = nn.Linear(current_size + hidden_size_avg * 2 + hidden_size_lag * 3 + hidden_size_vs_oppo +
                            hidden_oppo * 2, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, hidden_size3)
        self.l4 = nn.Linear(hidden_size3, hidden_size4)
        self.l5 = nn.Linear(hidden_size4, hidden_size5)
        self.l6 = nn.Linear(hidden_size5, hidden_size6)
        self.l7 = nn.Linear(hidden_size6, hidden_size7)
        self.l8 = nn.Linear(hidden_size7, hidden_size8)
        self.l9 = nn.Linear(hidden_size8, hidden_size9)
        self.out = nn.Linear(hidden_size9, out_size)

    def forward(self, X):
        x_current = X[:, :, :2]
        x_avg60 = X[:, :, 2:18]
        x_avg30 = X[:, :, 18:34]
        x_lag1 = X[:, :, 34:41]
        x_lag2 = X[:, :, 41:48]
        x_lag3 = X[:, :, 48:55]
        x_vs_oppo = X[:, :, 55:67]
        x_oppo_60 = X[:, :, 67:81]
        x_oppo_30 = X[:, :, 81:95]
        out_avg60 = self.avg(x_avg60)
        out_avg30 = self.avg(x_avg30)
        out_lag1 = self.lag(x_lag1)
        out_lag2 = self.lag(x_lag2)
        out_lag3 = self.lag(x_lag3)
        out_vs_oppo = self.vs_oppo(x_vs_oppo)
        out_oppo_30 = self.oppo(x_oppo_30)
        out_oppo_60 = self.oppo(x_oppo_60)
        out_avg60 = self.leaky_relu(out_avg60)
        out_avg30 = self.leaky_relu(out_avg30)
        out_lag1 = self.leaky_relu(out_lag1)
        out_lag2 = self.leaky_relu(out_lag2)
        out_lag3 = self.leaky_relu(out_lag3)
        out_vs_oppo = self.leaky_relu(out_vs_oppo)
        out_oppo_30 = self.leaky_relu(out_oppo_30)
        out_oppo_60 = self.leaky_relu(out_oppo_60)
        combined = torch.cat((x_current, out_avg60, out_avg30, out_lag1, out_lag2, out_lag3,
                              out_vs_oppo, out_oppo_60, out_oppo_30), 2)
        out = self.l1(combined)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.l3(out)
        out = self.leaky_relu(out)
        out = self.l4(out)
        out = self.leaky_relu(out)
        out = self.l5(out)
        out = self.leaky_relu(out)
        out = self.l6(out)
        out = self.leaky_relu(out)
        out = self.l7(out)
        out = self.leaky_relu(out)
        out = self.l8(out)
        out = self.leaky_relu(out)
        out = self.l9(out)
        out = self.leaky_relu(out)
        out = self.out(out)
        # no activation and no softmax at the end
        return out
def calculate_metrics(y_true,y_pred):
    result_metrics = {'mae': mean_absolute_error(y_true, y_pred),
                      'rmse': mean_squared_error(y_true, y_pred) ** 0.5,
                      'r2': r2_score(y_true, y_pred),
                      'max': max_error(y_true, y_pred)}
    print("RF max error:              ", result_metrics["max"])
    print("Mean Absolute Error:       ", result_metrics["mae"])
    print("Root Mean Squared Error:   ", result_metrics["rmse"])
    print("R^2 Score:                 ", result_metrics["r2"])
    return result_metrics
PlayerData = pd.read_csv('H:/python project 1/NBA data/games and player from 2004 to '
                         '2020/player_prediction_data_feature_selection_and_seconds.csv')
PlayerData = PlayerData.iloc[:, 5:]
X, y_ori = ew.feature_label_split(PlayerData, 'PLUS_MINUS')
scaler = ew.get_scaler('minmax')
X = scaler.fit_transform(X)
y = scaler.fit_transform(y_ori)
X = torch.Tensor(X)
y = torch.Tensor(y)
test = TensorDataset(X, y)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False)
loaded_model = torch.load("model1.pth")
predictions = []
values = []
for x_test, y_test in test_loader_one:
    x_test = x_test.view([1, 1, 95]).to(device)
    y_test = y_test.view([1, 1, 1]).to(device)
    loaded_model.eval()
    yhat = loaded_model(x_test)
    yhat = yhat.view([1])
    predictions.append(yhat.to(device).detach().numpy())

predictions = scaler.inverse_transform(predictions)
values = y_ori.to_numpy()
result_matrix = calculate_metrics(values,predictions)
resultCSVPath = r'H:/python project 1/NBA data/games and player from 2004 to 2020/predictions_player_PMs.csv'
pd.DataFrame(predictions).to_csv(resultCSVPath,index = False,na_rep = 0)

