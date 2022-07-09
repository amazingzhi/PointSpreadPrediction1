from sklearn.svm import SVR
import pandas as pd
import example_web as ew
from joblib import dump

models_dir = 'models_testing'
predictions_dir = 'predictions_testing'

train_path = '../data/games and player from 2004 to 2020/game_player_prediction_data/feature_selected_train.csv'
year_to_be_test = '2019'
target_col = 'pointspread'
col_to_drop = ['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'GAME_DATE_EST', 'pointspread']
transform = 'minmax'  # 'minmax', 'standard', "maxabs", "robust".
scaler = ew.get_scaler(transform)
data_pre = 'feature_selected'  # 'feature_selected', 'PCA'
resultCSVPath = f'{predictions_dir}/{data_pre}_SVM_{transform}.csv'

Train_X, Train_y, Test_X, Test_Y = ew.read_data_select_year_test_MLmodels(train_path=train_path, year_to_be_test=year_to_be_test,
                                                                  target_col=target_col, col_to_drop=col_to_drop,
                                                    scaler=scaler, data_pre=data_pre)

clf = SVR(kernel='poly', C=10)
clf.fit(Train_X, Train_y)
y_true, y_pred = Test_Y, clf.predict(Test_X)
result_matrix = ew.calculate_metrics_true_pred(y_true, y_pred)
result_matrix
dump(clf, f'{models_dir}/{data_pre}_SVM_{transform}_model.joblib')
pd.DataFrame(y_pred).to_csv(resultCSVPath, index=False, na_rep=0)