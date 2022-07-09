from sklearn.ensemble import RandomForestRegressor as RFR
import pandas as pd
import example_web as ew
from sklearn import preprocessing
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, r2_score, mean_squared_error
from joblib import dump, load

Preprocessing_Type = 'MinMaxScaler'  # 'StandardScaler', 'MinMaxScaler'
Type = 'original'  # 'original', 'feature_selection', 'PCA'
File_Path = 'data/games and player from 2004 to ' \
                '2020/player_prediction_data/player_prediction_data_feature_selection_and_seconds.csv '
col_to_drop = ['GAME_ID', 'PLAYER_ID', 'TEAM_ID', 'OPPOID', 'GAME_DATE_EST', 'PLUS_MINUS']
target_col = 'PLUS_MINUS'

df = pd.read_csv(File_Path)

X_train, Y_train = ew.feature_label_split(df=df,target_col=target_col,col_to_drop=col_to_drop,to_numpy=True)

# feature preprocessing
if Preprocessing_Type == 'StandardScaler':
    X_train = preprocessing.StandardScaler().fit_transform(X_train)

elif Preprocessing_Type == 'MinMaxScaler':
    X_train = preprocessing.MinMaxScaler().fit_transform(X_train)

clf = RFR(n_estimators=3, max_depth=3, min_samples_split=4,max_features='auto', min_samples_leaf=8,
                                     criterion='absolute_error', n_jobs=-1, verbose=10)
clf.fit(X_train, Y_train)
y_true, y_pred = Y_train, clf.predict(X_train)
print('Root Mean Squared Error:' + str(mean_squared_error(y_true, y_pred) ** 0.5))
print('max_error:' + str(max_error(y_true, y_pred)))
print('mean_absolute_error:' + str(mean_absolute_error(y_true, y_pred)))
print('r2_score:' + str(r2_score(y_true, y_pred)))

# save models and predictions
if Type == 'original':
    dump(clf, f'player_PM_prediction/models/original_RF_{Preprocessing_Type}_model.joblib')  # save model
    # clf = load('filename.joblib') load model
    print()
    # y_pred to csv
    resultCSVPath = f'player_PM_prediction/predictions/original_RF_{Preprocessing_Type}_prediction.csv'
    y_pred = pd.DataFrame(y_pred)
    y_pred.to_csv(resultCSVPath, index=False, na_rep=0)
# elif Type == 'feature_selection':
#     dump(clf, 'player_PM_prediction/models/feature_selection' + '_' + score + '_' + Model_Name + '_' + Preprocessing_Type + '_model.joblib')  # save model
#     # clf = load('filename.joblib') load model
#     print()
#     # y_pred to csv
#     resultCSVPath = f'player_PM_prediction/predictions/feature_selection_{Model_Name}_{score}_{Preprocessing_Type}_prediction.csv'
#     y_pred = pd.DataFrame(y_pred)
#     y_pred.to_csv(resultCSVPath, index=False, na_rep=0)
# elif Type == 'PCA':
#     dump(clf, 'player_PM_prediction/models/PCA' + '_' + score + '_' + Model_Name + '_' + Preprocessing_Type + '_model.joblib')  # save model
#     # clf = load('filename.joblib') load model
#     print()
#     # y_pred to csv
#     resultCSVPath = f'player_PM_prediction/predictions/PCA_{Model_Name}_{score}_{Preprocessing_Type}_prediction.csv'
#     y_pred = pd.DataFrame(y_pred)
#     y_pred.to_csv(resultCSVPath, index=False, na_rep=0)