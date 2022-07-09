## import libraries
import torch
import holidays
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import torch.optim as optim
from datetime import datetime
from datetime import date
from torch.utils.tensorboard import SummaryWriter
import shutil
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import Dataset, TensorDataset, DataLoader, TensorDataset, random_split, SubsetRandomSampler, \
    ConcatDataset
import numpy as np
import pandas as pd
import os
import glob
import csv
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
# uncomment if you want to create directory checkpoint, best_model!
# mkdir checkpoint best_model!
## see if cuda available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

class ProfitSimulation:
    """ProfitsSimulation is a a helper class that Simulates profits.
    Attributes:
            prediction_path: where you save all of your predictions in a file
            test_path: where you save your test dataset
            ori_path: where you save your original dataset
            bet_path: where you save your betting data
            games_all: how many games to select to simulate profits from all games
            games_one: how many games to select to simulate profits from one game per day
            new_columns_TF: column names for recording True or False for each algorithm
            new_columns_AS: column names for recording Accumulate Sum for each algorithm
            b: is the proportion of the bet gained with a win. E.g. If betting $10 on a 2-to-1 odds bet, (upon win you
                are returned $30, winning you $20), then {\displaystyle b=\$20/\$10=2.0}{\displaystyle b=\$20/\$10=2.0}.


    Methods:
            merge_all_predictions_to_a_DF: the predictions are csv files in a folder. Each file means predictions from
                one algorithm. So this method is for merge them into one Data Frame. If you already have a merged file,
                don't apply this method.
            data_cleaning: After I merge, I found two observations are 0s. So I remove these two observations. If you
                don't need to clean your data, don't apply this method.
            add_gameid_to_df: Because I need to match our prediction to the real game outcomes and betting lines, I have
                to give each prediction a game id so that it can be matched.
            import_betting_data_and_clean: import betting lines and drop irrelevant columns. And then remove observations
                that have same gameid which comes from other bookies. We only need one bookie for one game.
            merge_betting_and_prediction_by_gameid: match betting data and prediction data together by game ids.
            add_always_bet_home_away: add two columns which always bet home team or away team record as True or False
                for comparision with other algorithm.
            prediction: calculate if prediction is right or wrong and record as True or False.
            add_t_or_f_algorithms: loop over each algorithm and calculate if this algorithm for each prediction is right
                or wrong and record as True or False.
            best_bet: find best bet on each day for one algorithm.
            create_dic_records_best_bet: build a dictionary carries all best_bet for all algorithms.
            kelly_fraction: calculate kelly fraction by probability and odds.
            kelly_accumulative_sum: calculate one accumulative sum by kelly criterion.
            accumulative_sum: records all accumulative sums each day for one algorithm.
            add_accumulative_sum_to_df: add accumulative sums to Pandas or Dict.
            plot_accumulative_sum: plot accumulative sums.
            prediction_comparision_plot: plot all predictions with actual values.
        """

    def __init__(self, prediction_path: str, ori_path: str, bet_path: str, dow_path: str, sp500_path: str, BTC_path: str, commodity_path: str,
                 games_all_simple: int, games_all_kelly: int, games_one: int, new_columns_TF: list[str], new_columns_AS: list[str],
                 b_pointspread: float, initial_stake: int) -> 'ProfitSimulation object':
        self.prediction_path = prediction_path
        self.ori_path = ori_path
        self.bet_path = bet_path
        self.dow_path = dow_path
        self.sp500_path = sp500_path
        self.BTC_path = BTC_path
        self.commodity_path = commodity_path
        self.games_all_simple = games_all_simple
        self.games_all_kelly = games_all_kelly
        self.games_one = games_one
        self.new_columns_TF = new_columns_TF
        self.new_columns_AS = new_columns_AS
        self.b_pointspread = b_pointspread
        self.initial_stake = initial_stake

    def merge_all_predictions_to_a_DF(self) -> 'Pandas':
        all_files = glob.glob(os.path.join(self.prediction_path, "*.csv"))
        data_dic = {}
        for file in all_files:
            with open(file) as file_object:
                contents = file_object.read()
                data = contents.split('\n')
                temp = file.split('\\')[-1]
                column = temp.split('.')[0]
                data_dic[column] = data
        df = pd.DataFrame.from_dict(data_dic)
        return df

    def prediction_data_cleaning(self, df: 'Pandas') -> 'Pandas':
        df = df[1:]
        df = df[:-1]
        return df

    def add_gameid_and_date_to_df(self, df: 'Pandas') -> 'Pandas':
        ori = pd.read_csv(self.ori_path)
        ori = ori[ori['GAME_DATE_EST'].str.contains('2019')]
        df['date'] = list(ori['GAME_DATE_EST'])
        df['GAME_ID'] = list(ori['GAME_ID'])
        df = df.set_index('GAME_ID')
        df.index = df.index.map(str)
        df.columns = ['DNN', 'DTR', 'LR', 'RF', 'SVM', 'date']
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        return df

    def import_betting_data_and_clean(self) -> 'Pandas':
        betting = pd.read_csv(self.bet_path)
        betting = betting.drop(columns=['away_id', 'home_id'])
        betting = betting.set_index('game_id')
        betting.index = betting.index.map(str)
        return betting

    def add_correct_bet_columns(self, betting: 'Pandas') -> 'Pandas':
        correct_bets_gamewinner = []
        correct_bets_pointspread = []
        pointspreads = []
        for gameid in betting.index:
            home_point = betting['result'][gameid].split('-')[1]
            away_point = betting['result'][gameid].split('-')[0]
            home_minus_away = int(home_point) - int(away_point)
            pointspreads.append(home_minus_away)
            correct_bet_gamewinner = 1 if home_minus_away >= 0 else 0
            correct_bets_gamewinner.append(correct_bet_gamewinner)
            correct_bet_pointspread = 1 if home_minus_away + int(betting['home_plus_or_minus'][gameid]) >= 0 else 0
            correct_bets_pointspread.append(correct_bet_pointspread)
        betting['correct_pointspread_bet'] = correct_bets_pointspread
        betting['correct_gamewinner_bet'] = correct_bets_gamewinner
        betting['pointspread'] = pointspreads
        return betting

    def merge_betting_and_prediction_by_gameid(self, df: 'Pandas', betting: 'Pandas') -> 'Pandas':
        df = pd.merge(df, betting, left_index=True, right_index=True)
        # some extra data cleaning work to do
        date_nz = df['date']
        home = df['home']
        away = df['away']
        df = df.drop(columns=['date', 'datetime_nz', 'home', 'away', 'datetime_nz', 'result'])
        df = df.astype('float')
        df['home'] = home
        df['away'] = away
        df['date'] = date_nz
        return df

    def add_always_bet_home_away(self, df: 'Pandas', games_all: int) -> 'Pandas':
        temp_home = []
        temp_away = []
        for index, row in df.iterrows():
            if row['correct_gamewinner_bet'] == 1:
                temp_home.append(True)
                temp_away.append(False)
            else:
                temp_home.append(False)
                temp_away.append(True)
        print(f'the percentage of Ture of bet_home is {temp_home[:games_all].count(True) / len(temp_home[:games_all])}')
        print(f'the percentage of Ture of bet_away is {temp_away[:games_all].count(True) / len(temp_away[:games_all])}')
        df['home_TF_gamewinner'] = temp_home
        df['away_TF_gamewinner'] = temp_away
        return df

    def add_always_bet_favorite(self, df: 'Pandas', games_all: int) -> 'Pandas':
        temp_favorite = []
        for index, row in df.iterrows():
            if row['odds_home'] < 0 and row['pointspread'] >= 0 or row['odds_home'] > 0 and row['pointspread'] < 0:
                temp_favorite.append(True)
            else:
                temp_favorite.append(False)
        print(f'the percentage of Ture of bet_favorite is {temp_favorite[:games_all].count(True) / len(temp_favorite[:games_all])}')
        df['favorite_TF_gamewinner'] = temp_favorite
        return df

    # calculate if prediction is right or wrong and record as 1 or 0.
    def prediction(self, row: 'row in Pandas', column: 'column in Pandas', type: str) -> bool:
        temp = column.split('_')
        algorithm = temp[0]
        if type == 'pointspread':
            if row[algorithm] + row['home_plus_or_minus'] < 0:
                h_or_a = 0
            else:
                h_or_a = 1
            if h_or_a == row['correct_pointspread_bet']:
                return True
            return False
        elif type == 'gamewinner':
            if row[algorithm] >= 0:
                h_or_a = 1
            else:
                h_or_a = 0
            if h_or_a == row['correct_gamewinner_bet']:
                return True
            return False
    def add_t_or_f_algorithms(self, df: 'pandas', games_all: int, type: str) -> 'pandas':
        for column in self.new_columns_TF:
            temp = []
            for index, row in df.iterrows():
                temp.append(self.prediction(row, column, type))
            df[column + '_' + type] = temp
            print(f'the percentage of Ture of {column}_{type} is {temp[:games_all].count(True) / len(temp[:games_all])}')
        return df

    def best_bet(self, df: 'Pandas', column: str, type: str) -> list[bool]:
        temp = column.split('_')
        algorithm = temp[0]
        dates = df.date.unique()
        TF = []
        for date in dates:
            temp_data = {}
            for index, row in df.iterrows():
                df_date = row['date']
                if date == df_date:
                    temp_data[index] = abs(row[algorithm] + row['home_plus_or_minus'])
            max_key = max(temp_data, key=temp_data.get)
            TF.append(df.loc[max_key][algorithm + '_TF_' + type])
        print(f'the percentage of Ture of {column} is {TF.count(True) / len(TF)}')
        return TF

    def create_df_records_best_bet(self, df: 'pandas', type: str) -> 'Pandas':
        best_bet_of_the_day = [column + '_best_' + type for column in self.new_columns_TF]
        dic_best = {}
        for column in best_bet_of_the_day:
            dic_best[column] = self.best_bet(df, column, type)
        dates = list(df.date.unique())
        dic_best['date'] = dates
        df_best = pd.DataFrame(dic_best)
        return df_best

    # kelly fraction calculation
    def kelly_fraction(self, p: float, b: float) -> float:
        f = p + (p - 1) / b
        return f

    def kelly_accumulative_sum(self, f: float, b: float, last_AS: float, TF: bool, decision: str) -> float:
        if decision == 'bet_nothing' or decision == 'bet_one':
            if TF == True:
                this_AS = last_AS * f * b + last_AS
                return this_AS
            else:
                this_AS = last_AS * (1 - f)
                return this_AS
        else:
            if TF == False:
                this_AS = last_AS * f * b + last_AS
                return this_AS
            else:
                this_AS = last_AS * (1 - f)
                return this_AS

    def kelly_condition(self, b_one: float, b_two: float, p_one: float, p_two: float) -> str:
        if b_one == p_two/p_one:
            return 'bet_nothing'
        elif b_one > p_two/p_one:
            return 'bet_one'
        else:
            if b_two > p_one/p_two:
                return 'bet_two'
            else:
                return 'bet_nothing'

    def f_b_calculation(self, decision: str, b_one: float, b_two: float, p_one: float, p_two: float) -> tuple[float, float]:
        if decision == 'bet_nothing':
            return 0, 0
        elif decision == 'bet_one':
            return p_one - p_two/b_one, b_one
        else:
            return p_two - p_one/b_two, b_two

    def accumulative_sum_simple(self, column: str, data: 'pandas' or dict, type: str) -> list[float]:
        temp = column.split('_')
        algorithm = temp[0] + '_TF_' + type
        acc_sum = []
        for index, row in data.iterrows():
            if not acc_sum:
                acc_sum.append(self.simple_accumulative_sum(0, row[algorithm], type, temp[0], row))
            else:
                acc_sum.append(self.simple_accumulative_sum(acc_sum[-1], row[algorithm], type, temp[0], row))
        return acc_sum


    def simple_accumulative_sum(self, last_AS: float, TF: bool, type: str, method: str, row: 'Pandas Series') -> float:
        if type == 'pointspread':
            if TF == True:
                this_AS = last_AS + self.b_pointspread
            else:
                this_AS = last_AS - 1
        elif type == 'gamewinner':
            if TF == True:
                odd_home = self.calculate_b_by_given_odd(row['odds_home'])
                odd_away = self.calculate_b_by_given_odd(row['odds_away'])
                if method == 'home':
                    odd = odd_home
                elif method == 'away':
                    odd = odd_away
                elif method == 'favorite':
                    odd = odd_home if odd_home <= odd_away else odd_away
                else:
                    odd = odd_home if row[method] >= 0 else odd_away
                this_AS = last_AS + odd
            else:
                this_AS = last_AS - 1
        return this_AS

    def accumulative_sum_kelly(self, column: str, data: 'pandas' or dict, all_not_best: bool, type: str) -> list[float]:
        temp = column.split('_')
        if all_not_best == True:
            algorithm = temp[0] + '_TF_' + type
        else:
            algorithm = temp[0] + '_TF_best_' + type
        p_one = data[algorithm].value_counts(normalize=True).array[0]
        p_two = 1 - p_one
        acc_sum = []
        for index, row in data.iterrows():
            if type == 'pointspread':
                b_one = self.b_pointspread
                b_two = self.b_pointspread
            elif type == 'gamewinner':
                if row[temp[0]] >= 0:
                    b_one = self.calculate_b_by_given_odd(row['odds_home'])
                    b_two = self.calculate_b_by_given_odd(row['odds_away'])
                else:
                    b_one = self.calculate_b_by_given_odd(row['odds_away'])
                    b_two = self.calculate_b_by_given_odd(row['odds_home'])
            decision = self.kelly_condition(b_one, b_two, p_one, p_two)
            f, b = self.f_b_calculation(decision, b_one, b_two, p_one, p_two)
            if not acc_sum:
                acc_sum.append(self.kelly_accumulative_sum(f, b, self.initial_stake, row[algorithm], decision))
            else:
                acc_sum.append(self.kelly_accumulative_sum(f, b, acc_sum[-1], row[algorithm], decision))
        return acc_sum

    def calculate_b_by_given_odd(self, odd: int) -> float:
        if odd >= 0:
            b = odd/100
        else:
            b = -100/odd
        return b

    def add_accumulative_sum_to_df(self, data: 'Pandas', all_not_best: bool, type: str, kelly: bool) -> 'Pandas':
        if kelly:
            columns = self.new_columns_AS[:-3]
            for column in columns:
                data[column + '_' + type] = self.accumulative_sum_kelly(column, data, all_not_best, type)
        else:
            if type == 'gamewinner':
                columns = self.new_columns_AS
            else:
                columns = self.new_columns_AS[:-3]
            for column in columns:
                data[column + '_' + type] = self.accumulative_sum_simple(column, data, type)
        return data

    def accumulative_sum_to_returns(self, df: 'Pandas') -> 'Pandas':
        date = df['date']
        df = df.drop(columns=['date','DNN_TF_best_pointspread', 'DTR_TF_best_pointspread', 'LR_TF_best_pointspread',
                              'RF_TF_best_pointspread', 'SVM_TF_best_pointspread'])
        columns = df.columns
        df.to_numpy()
        df = (df-100)/100
        df = pd.DataFrame(df, columns=columns)
        df['date'] = date
        return df

    def plot_accumulative_sum(self, df: 'Pandas', number_of_games: int, break_even_line: int) -> 'Figure':
        AS = {}
        AS['Deep_Neural_Networks_Point_Spread'] = df['DNN_AS_pointspread']
        AS['Decision_Tree_Point_Spread'] = df['DTR_AS_pointspread']
        AS['Linear_Regression_Point_Spread'] = df['LR_AS_pointspread']
        AS['Random_Forest_Point_Spread'] = df['RF_AS_pointspread']
        AS['Support_Vector_Point_Spread'] = df['SVM_AS_pointspread']
        AS['Deep_Neural_Networks_Game_Winner'] = df['DNN_AS_gamewinner']
        AS['Decision_Tree_Game_Winner'] = df['DTR_AS_gamewinner']
        AS['Linear_Regression_Game_Winner'] = df['LR_AS_gamewinner']
        AS['Random_Forest_Game_Winner'] = df['RF_AS_gamewinner']
        AS['Support_Vector_Machine_Game_Winner'] = df['SVM_AS_gamewinner']
        if number_of_games > 1000:
            AS['Bet_Home_Game_Winner'] = df['home_AS_gamewinner']
            AS['Bet_Away_Game_Winner'] = df['away_AS_gamewinner']
            AS['Bet_Favorite_Game_Winner'] = df['favorite_AS_gamewinner']
        for name in AS.keys():
            plt.plot(range(1, number_of_games + 1), AS[name][:number_of_games], label=name)
        plt.axhline(y=break_even_line, color='k', linestyle='solid')
        plt.title('Profits Simulation')
        plt.xlabel('Number of Games Bet')
        plt.ylabel('Profits')
        plt.legend()
        plt.show()

    def plot_accumulative_return(self, df: 'Pandas', fin_assets: bool, sports_assets: bool, log: bool):
        AR = {}
        if fin_assets:
            AR['Bit_Coin'] = df['btc']
            AR['Dow_Jones_Commodity_index'] = df['com']
            AR['Dow_Jones_Stock_Index'] = df['dow']
            AR['S&P500_Index'] = df['sp']
        if sports_assets:
            AR['Deep_Neural_Networks'] = df['DNN_AS_pointspread']
            AR['Support_Vector_Regressor'] = df['SVM_AS_pointspread']
            AR['Decision_Tree_Rrgression'] = df['DTR_AS_pointspread']
            AR['Linear_Regression'] = df['LR_AS_pointspread']
            AR['Random_FOrest_Regression'] = df['RF_AS_pointspread']
        x = df['date']
        for name in AR.keys():
            if log:
                plt.plot(x, np.log(AR[name]), label=name)
            else:
                plt.plot(x, AR[name], label=name)
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()
        plt.axhline(y=0, color='k', linestyle='solid')
        plt.title('Accumulative Returns Simulation')
        plt.xlabel('Dates in 2019')
        plt.ylabel('Accumulative Returns')
        plt.legend(fontsize=6)
        plt.show()

    def prediction_comparision_plot(self, df: 'pandas') -> 'Figure':
        PS = {}
        actual = df['pointspread']
        PS['Deep_Neural_Networks'] = df['DNN']
        PS['Decision_Tree'] = df['DTR']
        PS['Linear_Regression'] = df['LR']
        PS['Random_Forest'] = df['RF']
        PS['Support_Vector_Machine'] = df['SVM']
        PS_keys = list(PS.keys())
        for i in range(2):
            fig, axs = plt.subplots(2, 2, sharex=True)
            axs_indexes = [[row, column] for row in range(2) for column in range(2)]
            for index, axs_index in enumerate(axs_indexes):
                axs[axs_index[0], axs_index[1]].plot(range(1, actual.size +1), actual, '-', label='Actual')
                if i == 0:
                    axs[axs_index[0], axs_index[1]].plot(range(1, actual.size +1), PS[PS_keys[index]], '-', label=PS_keys[index])
                    axs[axs_index[0], axs_index[1]].set_title(f'Actual vs {PS_keys[index]}')
                else:
                    if index != 0:
                        axs[axs_index[0], axs_index[1]].plot(range(1, actual.size +1), PS[PS_keys[index + 4]], '-', label=PS_keys[index + 4])
                        axs[axs_index[0], axs_index[1]].set_title(f'Actual vs {PS_keys[index + 4]}')
                    else:
                        break
                axs[axs_index[0], axs_index[1]].axhline(y=0, color='k', linestyle='solid')
                axs[axs_index[0], axs_index[1]].legend()

            for ax in axs.flat:
                ax.set(xlabel='Number of Games', ylabel='Point Spreads')
                ax.label_outer()  # Hide x labels and tick labels for top plots and y ticks for right plots.
            if i == 1:
                fig.delaxes(axs[1,1])
            fig.show()

    def read_indexes_data(self) -> tuple['Pandas', 'Pandas', 'Pandas', 'Pandas']:
        btc = pd.read_csv(self.BTC_path)
        com = pd.read_csv(self.commodity_path)
        dow = pd.read_csv(self.dow_path)
        sp = pd.read_csv(self.sp500_path)
        return btc, com, dow, sp

    def dow_data_clean(self, df: 'Pandas') -> 'Pandas':
        month_mapping = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
            'Nov': 11, 'Dec': 12
        }
        dates = []
        for date in df['Date']:
            temp = date.split('-')
            new_date = f'{str(month_mapping[temp[1]])}/{temp[0]}/20{temp[2]}'
            dates.append(new_date)
        df['Date'] = dates
        df = df.drop(columns=['Open', 'High', 'Low', 'Vol.', 'Change %'])
        df['Price'] = df['Price'].apply(lambda x: float(x.split(',')[0] + x.split(',')[1]))
        return df

    def stocks_data_clean(self, btc: 'Pandas', com: 'Pandas', dow: 'Pandas', sp: 'Pandas') -> list('Pandas'):
        stocks = [btc, com, dow, sp]
        new_stocks = []
        for i, stock in enumerate(stocks):
            stock = stock[stock['Date'].str.contains('2019')]
            returns = []
            for price in stock['Price']:
                Return = (price - stock['Price'].iloc[0])/stock['Price'].iloc[0]
                returns.append(Return)
            stock[f'returns_{i}'] = returns
            stock = stock.drop(columns=['Price'])
            new_stocks.append(stock)
        return new_stocks

    def stocks_data_merge(self, stocks: list('Pandas')) -> 'Pandas':
        base_df = stocks[0]
        for i in range(1,4):
            base_df = base_df.merge(stocks[i], how='left', on='Date')
        base_df.columns = ['date','btc','com','dow','sp']
        base_df.fillna(method='ffill', inplace=True)
        base_df.fillna(value=0, inplace=True)
        dates = base_df['date']
        base_df['date'] = [dt.datetime.strptime(d, '%m/%d/%Y').date() for d in dates]
        return base_df
# read data
def read_data_select_year_test_MLmodels(train_path=None,year_to_be_test=None, target_col=None,
                               col_to_drop=None, scaler=None, data_pre=None):
    Train = pd.read_csv(train_path)
    Test = Train[Train['GAME_DATE_EST'].str.contains(year_to_be_test)]
    Train = Train[~Train['GAME_DATE_EST'].str.contains(year_to_be_test)]
    Train_X, Train_y_ori = feature_label_split(df=Train, target_col=target_col,
                                               col_to_drop=col_to_drop, to_numpy=True)
    Test_X, Test_y_ori = feature_label_split(df=Test, target_col=target_col, col_to_drop=col_to_drop, to_numpy=True)
    if data_pre == 'PCA' and isinstance(scaler, StandardScaler):
        pass
    else:
        Train_X = scaler.fit_transform(Train_X)
        Test_X = scaler.transform(Test_X)
    return Train_X, Train_y_ori, Test_X, Test_y_ori
def read_data_select_year_test(train_path=None,year_to_be_test=None, target_col=None,
                               col_to_drop=None, scaler_X=None, scaler_y=None, data_pre=None):
    Train = pd.read_csv(train_path)
    Test = Train[Train['GAME_DATE_EST'].str.contains(year_to_be_test)]
    Train = Train[~Train['GAME_DATE_EST'].str.contains(year_to_be_test)]
    Train_X, Train_y_ori = feature_label_split(df=Train, target_col=target_col,
                                               col_to_drop=col_to_drop, to_numpy=True)  # x, y split and data cleaning
    Test_X, Test_y_ori = feature_label_split(df=Test, target_col=target_col, col_to_drop=col_to_drop, to_numpy=True)
    if data_pre == 'PCA' and isinstance(scaler_X, StandardScaler):
        pass
    else:
        Train_X = scaler_X.fit_transform(Train_X)  # normalization
        Test_X = scaler_X.transform(Test_X)  # normalization
        # Train_y_ori = scaler_y.fit_transform(Train_y_ori)
        # Test_y_ori = scaler_y.transform(Test_y_ori)
    train_features = torch.Tensor(Train_X)  # df to tensor
    train_targets = torch.Tensor(Train_y_ori)  # y_train_arr
    train = TensorDataset(train_features, train_targets)  # tensor to dataloader
    test_features = torch.Tensor(Test_X)  # df to tensor
    test_targets = torch.Tensor(Test_y_ori)  # y_test_arr
    test = TensorDataset(test_features, test_targets)  # tensor to dataloader
    return train, test
def read_data(train_and_test=None, train_path=None, test_path=None, target_col=None, col_to_drop=None, scaler=None):
    if train_and_test:  # if two datasets train and test:
        Test = pd.read_csv(test_path)
        Test_X, Test_y_ori = feature_label_split(df=Test, target_col=target_col, col_to_drop=col_to_drop, to_numpy=False)
        X_test_arr = scaler.fit_transform(Test_X)  # normalization
        y_test_arr = scaler.fit_transform(Test_y_ori)
        test_features = torch.Tensor(X_test_arr)  # df to tensor
        test_targets = torch.Tensor(y_test_arr)
        test = TensorDataset(test_features, test_targets)  # tensor to dataloader
    Train = pd.read_csv(train_path)  # if only train:
    Train_X, Train_y_ori = feature_label_split(df=Train, target_col=target_col,
                                               col_to_drop=col_to_drop, to_numpy=False)  # x, y split and data cleaning
    X_train_arr = scaler.fit_transform(Train_X)  # normalization
    y_train_arr = scaler.fit_transform(Train_y_ori)
    train_features = torch.Tensor(X_train_arr)  # df to tensor
    train_targets = torch.Tensor(y_train_arr)
    train = TensorDataset(train_features, train_targets)  # tensor to dataloader
    if train_and_test:
        return train, test
    else:
        return train


## get one hot encoding for pandas dataframe
def onehot_encode_pd(df, cols):
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col)

    return pd.concat([df, dummies], axis=1).drop(columns=cols)


## find which day is holiday in US.
us_holidays = holidays.US()


def is_holiday(date):
    date = date.replace(hour=0)
    return 1 if (date in us_holidays) else 0


def add_holiday_col(df, holidays):
    return df.assign(is_holiday=df.index.to_series().apply(is_holiday))


## get X and y by given pandas df
def feature_label_split(df, target_col, col_to_drop, to_numpy):
    y = df[[target_col]]
    X = df.drop(columns=col_to_drop)
    if to_numpy:
        X = X.to_numpy()
        y = y.to_numpy()
    return X, y


def train_test_split1(df, target_col, test_ratio, col_to_drop, to_numpy):
    X, y = feature_label_split(df, target_col, col_to_drop, to_numpy)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    return X_train, X_test, y_train, y_test


## train validate test split
def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test


# X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_features, 'value', 0.2)

## standardization or normalization
def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()


def model_parameters(length=None, nodes_propotion=None, denominator_of_input=None):
    input_dim = length
    input_size = input_dim
    hidden_size1 = (input_dim // denominator_of_input) * nodes_propotion
    hidden_size2 = (input_dim // denominator_of_input) * nodes_propotion
    hidden_size3 = (input_dim // denominator_of_input) * nodes_propotion
    hidden_size4 = (input_dim // denominator_of_input) * nodes_propotion
    hidden_size5 = (input_dim // denominator_of_input) * nodes_propotion
    hidden_size6 = (input_dim // denominator_of_input) * nodes_propotion
    hidden_size7 = (input_dim // denominator_of_input) * nodes_propotion
    hidden_size8 = (input_dim // denominator_of_input) * nodes_propotion
    hidden_size9 = (input_dim // denominator_of_input) * nodes_propotion
    out_size = 1

    model_params = {
        'model_params1': {'input_size': input_size, 'hidden_size1': hidden_size1, 'out_size': out_size},
        'model_params2': {'input_size': input_size, 'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2,
                          'out_size': out_size},
        'model_params3': {'input_size': input_size, 'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2,
                          'hidden_size3': hidden_size3,
                          'out_size': out_size},
        'model_params4': {'input_size': input_size, 'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2,
                          'hidden_size3': hidden_size3,
                          'hidden_size4': hidden_size4, 'out_size': out_size},
        'model_params5': {'input_size': input_size, 'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2,
                          'hidden_size3': hidden_size3,
                          'hidden_size4': hidden_size4, 'hidden_size5': hidden_size5, 'out_size': out_size},
        'model_params6': {'input_size': input_size, 'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2,
                          'hidden_size3': hidden_size3,
                          'hidden_size4': hidden_size4, 'hidden_size5': hidden_size5, 'hidden_size6': hidden_size6,
                          'out_size': out_size},
        'model_params9': {'input_size': input_size, 'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2,
                          'hidden_size3': hidden_size3,
                          'hidden_size4': hidden_size4, 'hidden_size5': hidden_size5, 'hidden_size6': hidden_size6,
                          'hidden_size7': hidden_size7, 'hidden_size8': hidden_size8, 'hidden_size9': hidden_size9,
                          'out_size': out_size}
    }
    return model_params, input_dim


def training_parameters(learning_rates=None, batch_sizes=None, weight_decays=None, num_layers=None, nodes_propotions=None):
    training_params = dict(
        learning_rate=learning_rates,
        batch_size=batch_sizes,
        weight_decay=weight_decays,
        num_layers=num_layers,
        nodes_propotions=nodes_propotions
    )
    param_values = [v for v in training_params.values()]
    return param_values


def cross_validation(param_values=None, denominator_of_input=None, splits=None, train_data=None, n_epochs=None,
                     folds=None, data_pre=None, transform=None, runs_dir=None, model_dir=None):
    validate_loss_min = np.inf
    for run_id, (lr, batch_size, weight_decay, num_layer, nodes_propotion) in enumerate(product(*param_values)):
        print("parameter set id:", run_id + 1)
        # add tensorboard
        print(f'training parameters: learning rate-{lr} batch_size-{batch_size}, weight decay-{weight_decay}, num ' \
              f'layers-{num_layer}, nodes_propotion-{nodes_propotion}')
        model_params, input_dim = model_parameters(train_data.tensors[0].shape[1], nodes_propotion=nodes_propotion, denominator_of_input=denominator_of_input)  # model parameters
        valid_losses_by_given_params = []
        cross_validation_models = []
        cross_validation_optimizers = []
        cross_validation_opt = []
        for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train_data)))):
            model = get_model(f'dnn{num_layer[-1]}', model_params[num_layer])
            loss_fn = nn.L1Loss(reduction="mean")
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
            comment = f'learning_rate={lr} batch_size={batch_size} weight_decay={weight_decay} num_layers={num_layer} ' \
                      f'nodes_propotion={nodes_propotion} fold={fold+1}'
            tb = SummaryWriter(
                log_dir=f'{runs_dir}_{data_pre}_{transform}/learning_rate={lr} batch_size={batch_size} '
                        f'weight_decay={weight_decay} num_layers={num_layer} nodes_propotion={nodes_propotion} fold={fold+1}',
                comment=comment)  # Todo: after you change datasets, put
            # "tensorboard --logdir runs_original_{transform} ; runs_feature_selected_{transform} ; runs_PCA_{transform}";
            print('Fold {}'.format(fold + 1))
            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, sampler=train_sampler)
            val_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, sampler=test_sampler)

            valid_loss_by_given_params = opt.train(train_loader=train_loader, val_loader=val_loader,
                                                   checkpoint_path="checkpoint/current_checkpoint.pt",
                                                   best_model_path=f"{model_dir}/DNN_{data_pre}_{transform}_{fold+1}"
                                                                   f"_{lr}_{batch_size}_{weight_decay}_{num_layer}_{nodes_propotion}.pt",
                                                   tb=tb, valid_loss_min_input=np.Inf,
                                                   batch_size=batch_size, n_epochs=n_epochs,
                                                   n_features=input_dim, fold=fold, folds=folds)
            valid_losses_by_given_params.append(valid_loss_by_given_params)
            cross_validation_models.append(model)
            cross_validation_optimizers.append(optimizer)
            cross_validation_opt.append(opt)
        min_fold = np.argmin(valid_losses_by_given_params)
        min_fold_loss = np.min(valid_losses_by_given_params)
        print(f'the min loss of these {folds} folds is {min_fold_loss}, and its fold is {fold+1}.')
        mean_valid_loss_by_given_params = np.mean(valid_losses_by_given_params)
        if mean_valid_loss_by_given_params < validate_loss_min:
            print('There is a better set of parameters.')
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(validate_loss_min,
                                                                                            mean_valid_loss_by_given_params))
            best_model_path = f"{model_dir}/DNN_{data_pre}_{transform}_{min_fold+1}_{lr}_{batch_size}_{weight_decay}_{num_layer}_{nodes_propotion}.pt"  # todo
            validate_loss_min = mean_valid_loss_by_given_params
            best_parameters = f'learning_rate={lr} batch_size={batch_size} weight_decay={weight_decay} num_layers={num_layer} nodes_propotion={nodes_propotion}'
            best_model = cross_validation_models[min_fold]
            best_optimizer = cross_validation_optimizers[min_fold]
            best_opt = cross_validation_opt[min_fold]
        print("__________________________________________________________")
        comment = f'learning_rate={lr} batch_size={batch_size} weight_decay={weight_decay} num_layers={num_layer} ' \
                      f'nodes_propotion={nodes_propotion}'
        tb = SummaryWriter(
            log_dir=f'{runs_dir}_{data_pre}_{transform}/learning_rate={lr} batch_size={batch_size} '
                    f'weight_decay={weight_decay} num_layers={num_layer} nodes_propotion={nodes_propotion}',
            comment=comment)
        tb.add_hparams(
            {"lr": lr, "bsize": batch_size, "weight_decay": weight_decay, "layers": num_layer, "nodes_propotion": nodes_propotion},
            {
                "loss": mean_valid_loss_by_given_params,
            },
        )
        X, y = next(iter(train_loader))
        tb.add_graph(model, X)
    tb.close()
    print(best_parameters + ';   min loss: ' + str(validate_loss_min))
    return best_opt, best_model, best_optimizer, best_model_path


### DNN 1 layer
class DNNModel1(nn.Module):
    def __init__(self, input_size, hidden_size1, out_size):
        super(DNNModel1, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.out_size = out_size
        self.leaky_relu = nn.LeakyReLU()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.out = nn.Linear(hidden_size1, out_size)

    def forward(self, X):
        out = self.l1(X)
        out = self.leaky_relu(out)
        out = self.out(out)
        # no activation and no softmax at the end
        return out


### DNN 2 layers
class DNNModel2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, out_size):
        super(DNNModel2, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.out_size = out_size
        self.leaky_relu = nn.LeakyReLU()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.out = nn.Linear(hidden_size2, out_size)

    def forward(self, X):
        out = self.l1(X)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.out(out)
        # no activation and no softmax at the end
        return out


### DNN 3 layers
class DNNModel3(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, out_size):
        super(DNNModel3, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.out_size = out_size
        self.leaky_relu = nn.LeakyReLU()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, hidden_size3)
        self.out = nn.Linear(hidden_size3, out_size)

    def forward(self, X):
        out = self.l1(X)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.l3(out)
        out = self.leaky_relu(out)
        out = self.out(out)
        # no activation and no softmax at the end
        return out


### DNN 4 layers
class DNNModel4(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, out_size):
        super(DNNModel4, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.out_size = out_size
        self.leaky_relu = nn.LeakyReLU()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, hidden_size3)
        self.l4 = nn.Linear(hidden_size3, hidden_size4)
        self.out = nn.Linear(hidden_size4, out_size)

    def forward(self, X):
        out = self.l1(X)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.l3(out)
        out = self.leaky_relu(out)
        out = self.l4(out)
        out = self.leaky_relu(out)
        out = self.out(out)
        # no activation and no softmax at the end
        return out


### DNN 5 layers
class DNNModel5(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, out_size):
        super(DNNModel5, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.hidden_size5 = hidden_size5
        self.out_size = out_size
        self.leaky_relu = nn.LeakyReLU()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, hidden_size3)
        self.l4 = nn.Linear(hidden_size3, hidden_size4)
        self.l5 = nn.Linear(hidden_size4, hidden_size5)
        self.out = nn.Linear(hidden_size5, out_size)

    def forward(self, X):
        out = self.l1(X)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.l3(out)
        out = self.leaky_relu(out)
        out = self.l4(out)
        out = self.leaky_relu(out)
        out = self.l5(out)
        out = self.leaky_relu(out)
        out = self.out(out)
        # no activation and no softmax at the end
        return out


### DNN 6 layers
class DNNModel6(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6,
                 out_size):
        super(DNNModel6, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.hidden_size5 = hidden_size5
        self.hidden_size6 = hidden_size6
        self.out_size = out_size
        self.leaky_relu = nn.LeakyReLU()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, hidden_size3)
        self.l4 = nn.Linear(hidden_size3, hidden_size4)
        self.l5 = nn.Linear(hidden_size4, hidden_size5)
        self.l6 = nn.Linear(hidden_size5, hidden_size6)
        self.out = nn.Linear(hidden_size6, out_size)

    def forward(self, X):
        out = self.l1(X)
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
        out = self.out(out)
        # no activation and no softmax at the end
        return out


### DNN 9 layers
class DNNModel9(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6,
                 hidden_size7, hidden_size8, hidden_size9,
                 out_size):
        super(DNNModel9, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.hidden_size5 = hidden_size5
        self.hidden_size6 = hidden_size6
        self.hidden_size7 = hidden_size7
        self.hidden_size8 = hidden_size8
        self.hidden_size9 = hidden_size9
        self.out_size = out_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.LeakyReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.LeakyReLU(),
            nn.Linear(hidden_size3, hidden_size4),
            nn.LeakyReLU(),
            nn.Linear(hidden_size4, hidden_size5),
            nn.LeakyReLU(),
            nn.Linear(hidden_size5, hidden_size6),
            nn.LeakyReLU(),
            nn.Linear(hidden_size6, hidden_size7),
            nn.LeakyReLU(),
            nn.Linear(hidden_size7, hidden_size8),
            nn.LeakyReLU(),
            nn.Linear(hidden_size8, hidden_size9),
            nn.LeakyReLU(),
            nn.Linear(hidden_size9, out_size)
        )
        # self.leaky_relu = nn.LeakyReLU()
        # self.l1 = nn.Linear(input_size, hidden_size1)
        # self.l2 = nn.Linear(hidden_size1, hidden_size2)
        # self.l3 = nn.Linear(hidden_size2, hidden_size3)
        # self.l4 = nn.Linear(hidden_size3, hidden_size4)
        # self.l5 = nn.Linear(hidden_size4, hidden_size5)
        # self.l6 = nn.Linear(hidden_size5, hidden_size6)
        # self.l7 = nn.Linear(hidden_size6, hidden_size7)
        # self.l8 = nn.Linear(hidden_size7, hidden_size8)
        # self.l9 = nn.Linear(hidden_size8, hidden_size9)
        # self.out = nn.Linear(hidden_size9, out_size)

    def forward(self, X):
        out = self.layers(X)


        # no activation and no softmax at the end
        return out


### RNN model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """The __init__ method that initiates an RNN instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # RNN layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out


### LSTM model
class LSTMModel(nn.Module):
    """LSTMModel class extends nn.Module class and works as a constructor for LSTMs.

       LSTMModel class initiates a LSTM module based on PyTorch's nn.Module class.
       It has only two methods, namely init() and forward(). While the init()
       method initiates the model with the given input parameters, the forward()
       method defines how the forward propagation needs to be calculated.
       Since PyTorch automatically defines back propagation, there is no need
       to define back propagation method.

       Attributes:
           hidden_dim (int): The number of nodes in each layer
           layer_dim (str): The number of layers in the network
           lstm (nn.LSTM): The LSTM model constructed with the input parameters.
           fc (nn.Linear): The fully connected layer to convert the final state
                           of LSTMs to our desired output shape.

    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """The __init__ method that initiates a LSTM instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out


### GRU
class GRUModel(nn.Module):
    """GRUModel class extends nn.Module class and works as a constructor for GRUs.

       GRUModel class initiates a GRU module based on PyTorch's nn.Module class.
       It has only two methods, namely init() and forward(). While the init()
       method initiates the model with the given input parameters, the forward()
       method defines how the forward propagation needs to be calculated.
       Since PyTorch automatically defines back propagation, there is no need
       to define back propagation method.

       Attributes:
           hidden_dim (int): The number of nodes in each layer
           layer_dim (str): The number of layers in the network
           gru (nn.GRU): The GRU model constructed with the input parameters.
           fc (nn.Linear): The fully connected layer to convert the final state
                           of GRUs to our desired output shape.

    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """The __init__ method that initiates a GRU instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out


### choose model to do RNN
def get_model(model, model_params):
    models = {
        "rnn": RNNModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
        "dnn1": DNNModel1,
        "dnn2": DNNModel2,
        "dnn3": DNNModel3,
        "dnn4": DNNModel4,
        "dnn5": DNNModel5,
        "dnn6": DNNModel6,
        "dnn9": DNNModel9
    }
    return models.get(model.lower())(**model_params)


# saving function
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


# loading function
def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


### Optimization is a helper class that allows training, validation, prediction
class Optimization:
    """Optimization is a helper class that allows training, validation, prediction.

    Optimization is a helper class that takes model, loss function, optimizer function
    learning scheduler (optional), early stopping (optional) as inputs. In return, it
    provides a framework to train and validate the models, and to predict future values
    based on the models.

    Attributes:
        model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
        loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
        optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        train_losses (list[float]): The loss values from the training
        val_losses (list[float]): The loss values from the validation
        last_epoch (int): The number of epochs that the models is trained
    """

    def __init__(self, model, loss_fn, optimizer):
        """
        Args:
            model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
            loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
            optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        """The method train_step completes one step of training.

        Given the features (x) and the target values (y) tensors, the method completes
        one step of the training. First, it activates the train mode to enable back prop.
        After generating predicted values (yhat) by doing forward propagation, it calculates
        the losses by using the loss function. Then, it computes the gradients by doing
        back propagation and updates the weights by calling step() function.

        Args:
            x (torch.Tensor): Tensor for features to train one step
            y (torch.Tensor): Tensor for target values to calculate losses

        """
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, checkpoint_path, best_model_path, tb, valid_loss_min_input=np.Inf,
              batch_size=64, n_epochs=50, n_features=1, fold=None, folds=None):
        """The method train performs the model training

        The method takes DataLoaders for training and validation datasets, batch size for
        mini-batch training, number of epochs to train, and number of features as inputs.
        Then, it carries out the training by iteratively calling the method train_step for
        n_epochs times. If early stopping is enabled, then it  checks the stopping condition
        to decide whether the training needs to halt before n_epochs steps. Finally, it saves
        the model in a designated file path.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader that stores training data
            val_loader (torch.utils.data.DataLoader): DataLoader that stores validation data
            batch_size (int): Batch size for mini-batch training
            n_epochs (int): Number of epochs, i.e., train steps, to train
            n_features (int): Number of feature columns

        """
        # initialize tracker for minimum validation loss
        valid_loss_min = valid_loss_min_input

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                # x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                # y_batch = y_batch.view([batch_size, 1, 1]).to(device)
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    # x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    # y_val = y_val.view([batch_size, 1, 1]).to(device)
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
            tb.add_scalar("Training Loss", training_loss, epoch)
            tb.add_scalar("Validation Loss", validation_loss, epoch)

            if (epoch <= 10) | (epoch % 1 == 0):
                print(
                    f"[{fold+1}/{folds}][{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )
            # create checkpoint variable and add important data
            checkpoint = {
                'fold': fold + 1,
                'epoch': epoch + 1,
                'valid_loss_min': validation_loss,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            # save checkpoint
            save_ckp(checkpoint, False, checkpoint_path, best_model_path)

            ## TODO: save the model if validation loss has decreased
            if validation_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                validation_loss))
                # save checkpoint as best model
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                valid_loss_min = validation_loss
        return valid_loss_min

    def evaluate(self, test_loader, model_ori, optimizer_ori, best_model_path, batch_size=1, n_features=1):
        """The method evaluate performs the model evaluation

        The method takes DataLoaders for the test dataset, batch size for mini-batch testing,
        and number of features as inputs. Similar to the model validation, it iteratively
        predicts the target values and calculates losses. Then, it returns two lists that
        hold the predictions and the actual values.

        Note:
            This method assumes that the prediction from the previous step is available at
            the time of the prediction, and only does one-step prediction into the future.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader that stores test data
            batch_size (int): Batch size for mini-batch training
            n_features (int): Number of feature columns

        Returns:
            list[float]: The values predicted by the model
            list[float]: The actual values in the test set.

        """
        # define checkpoint saved path
        ckp_path = best_model_path
        # load the saved checkpoint
        model, optimizer, start_epoch, valid_loss_min = load_ckp(ckp_path, model_ori, optimizer_ori)
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                # x_test = x_test.view([batch_size, -1, n_features]).to(device)
                # y_test = y_test.view([batch_size, 1, 1]).to(device)
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                # self.model.eval()
                # yhat = self.model(x_test)
                model.eval()
                yhat = model(x_test)
                yhat = yhat.view([1])  # comment when use for format prediction function in example_web.py
                predictions.append(yhat.to(device).detach().numpy())
                y_test = y_test.view([1])  # comment when use for format prediction function in example_web.py
                values.append(y_test.to(device).detach().numpy())

        return predictions, values

    def plot_losses(self):
        """The method plots the calculated loss values for training and validation
        """
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()


# ## Training the model
# import torch.optim as optim
#
# input_dim = len(X_train.columns)
# output_dim = 1
# hidden_dim = 64
# layer_dim = 3
# batch_size = 64
# dropout = 0.2
# n_epochs = 50
# learning_rate = 1e-3
# weight_decay = 1e-6
#
# model_params = {'input_dim': input_dim,
#                 'hidden_dim' : hidden_dim,
#                 'layer_dim' : layer_dim,
#                 'output_dim' : output_dim,
#                 'dropout_prob' : dropout}
#
# model = get_model('lstm', model_params)
#
# loss_fn = nn.MSELoss(reduction="mean")
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#
#
# opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
# opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
# opt.plot_losses()
#
# predictions, values = opt.evaluate(
#     test_loader_one,
#     batch_size=1,
#     n_features=input_dim
# )

## format prediction
def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def format_predictions(predictions, values, df_test, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result


# df_result = format_predictions(predictions, values, X_test, scaler)
# df_result

## Calculating error metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error


def calculate_metrics_from_df(df):
    result_metrics = {'mae': mean_absolute_error(df.value, df.prediction),
                      'rmse': mean_squared_error(df.value, df.prediction) ** 0.5,
                      'r2': r2_score(df.value, df.prediction),
                      'max': max_error(df.value, df.prediction)}
    print("RF max error:              ", result_metrics["max"])
    print("Mean Absolute Error:       ", result_metrics["mae"])
    print("Root Mean Squared Error:   ", result_metrics["rmse"])
    print("R^2 Score:                 ", result_metrics["r2"])
    return result_metrics


def calculate_metrics_true_pred(y_true, y_pred):
    result_metrics = {'mae': mean_absolute_error(y_true, y_pred),
                      'rmse': mean_squared_error(y_true, y_pred) ** 0.5,
                      'r2': r2_score(y_true, y_pred),
                      'max': max_error(y_true, y_pred)}
    print("RF max error:              ", result_metrics["max"])
    print("Mean Absolute Error:       ", result_metrics["mae"])
    print("Root Mean Squared Error:   ", result_metrics["rmse"])
    print("R^2 Score:                 ", result_metrics["r2"])
    return result_metrics


# result_metrics = calculate_metrics(df_result)

## Generating baseline predictions(linear regression benchmark)
from sklearn.linear_model import LinearRegression


def build_baseline_model(df, test_ratio, target_col):
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    result = pd.DataFrame(y_test)
    result["prediction"] = prediction
    result = result.sort_index()

    return result

# df_baseline = build_baseline_model(df_features, 0.2, 'value')
# baseline_metrics = calculate_metrics(df_baseline)

# ## Visualizing the predictions
# import plotly.offline as pyo
# import plotly.graph_objs as go
# from plotly.offline import iplot
#
#
# def plot_predictions(df_result, df_baseline):
#     data = []
#
#     value = go.Scatter(
#         x=df_result.index,
#         y=df_result.value,
#         mode="lines",
#         name="values",
#         marker=dict(),
#         text=df_result.index,
#         line=dict(color="rgba(0,0,0, 0.3)"),
#     )
#     data.append(value)
#
#     baseline = go.Scatter(
#         x=df_baseline.index,
#         y=df_baseline.prediction,
#         mode="lines",
#         line={"dash": "dot"},
#         name='linear regression',
#         marker=dict(),
#         text=df_baseline.index,
#         opacity=0.8,
#     )
#     data.append(baseline)
#
#     prediction = go.Scatter(
#         x=df_result.index,
#         y=df_result.prediction,
#         mode="lines",
#         line={"dash": "dot"},
#         name='predictions',
#         marker=dict(),
#         text=df_result.index,
#         opacity=0.8,
#     )
#     data.append(prediction)
#
#     layout = dict(
#         title="Predictions vs Actual Values for the dataset",
#         xaxis=dict(title="Time", ticklen=5, zeroline=False),
#         yaxis=dict(title="Value", ticklen=5, zeroline=False),
#     )
#
#     fig = dict(data=data, layout=layout)
#     iplot(fig)
#
#
# # Set notebook mode to work in offline
# pyo.init_notebook_mode()
#
# plot_predictions(df_result, df_baseline)
