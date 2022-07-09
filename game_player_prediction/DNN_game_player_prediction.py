# import libraries
import torch
import pandas as pd
from torch.utils.data import DataLoader
import example_web as ew
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

""" 
ToDo YOU HAVE TO CREATE 'checkpoint' and 'models' and 'predictions' folders in your working directory
Todo: after you change datasets, put "tensorboard --logdir runs_feature_selected ; runs_PCA" into terminal and
    change working directory. after you have run the program, put "tensorboard --logdir runs_original ; runs_feature_selected ; runs_PCA" into terminal again to see your results in tensorboard. 
todo : change variables with comments after codes when you change datasets"""
# global parameters set up
    # training parameters
n_epochs = 99
k = 3
shuffle = True
random_state = 42
splits = KFold(n_splits=k,shuffle=shuffle,random_state=random_state)
    # model parameters
learning_rates=[1e-3, 1e-2]  #1e-3, 1e-2
batch_sizes=[16, 32, 64]  #
weight_decays=[1e-4,1e-6]  #
nodes_propotions=[1, 2, 3]  #
denominator_of_input=len(nodes_propotions)
num_layers = ['model_params1', 'model_params2', 'model_params3']  # , 'model_params2', 'model_params3', 'model_params4', 'model_params5',
    # tensorboard
tb = SummaryWriter()
    #read data parameters
train_and_test = True  # True
data_pre = 'PCA'  # 'feature_selected', 'PCA'
train_path = '../data/games and player from 2004 to 2020/game_player_prediction_data/pca_train.csv'
model_dir = 'models_2019_test'
predictions_dir = 'predictions_2019_test'
runs_dir = 'runs_2019'
#'../data/games and player from 2004 to 2020/game_player_prediction_data/feature_selected_train.csv'
#'../data/games and player from 2004 to 2020/game_player_prediction_data/pca_train.csv'
year_to_be_test = '2019'  # the year to be test dataset.
# test_path = '../data/games and player from 2004 to 2020/game_player_prediction_data/pca_test.csv'
# '../data/games and player from 2004 to 2020/game_player_prediction_data/feature_selected_test.csv'
# '../data/games and player from 2004 to 2020/game_player_prediction_data/pca_test.csv'
target_col = 'pointspread'
col_to_drop = ['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'GAME_DATE_EST', 'pointspread']
transform = 'minmax'  # 'minmax', 'standard', "maxabs", "robust".
scaler_X = ew.get_scaler(transform)
scaler_y = ew.get_scaler(transform)
resultCSVPath = f'{predictions_dir}/{data_pre}_DNN_{transform}.csv'  # predictions/{data_pre}_DNN_{transform}.csv

## see if cuda available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

def main():
    if train_and_test:
        train, test = ew.read_data_select_year_test(train_path=train_path, year_to_be_test=year_to_be_test,
                                                                  target_col=target_col, col_to_drop=col_to_drop,
                                                    scaler_X=scaler_X, scaler_y=scaler_y, data_pre=data_pre)
    else:
        train = ew.read_data(train_and_test=train_and_test, train_path=train_path, test_path=test_path,
                                      target_col=target_col,col_to_drop=col_to_drop,scaler=scaler)

    # training parameters
    param_values = ew.training_parameters(learning_rates=learning_rates,batch_sizes=batch_sizes,
                                          weight_decays=weight_decays,num_layers=num_layers,nodes_propotions=nodes_propotions)


    # training with cross validation
    best_opt, best_model, best_optimizer, best_model_path = ew.cross_validation(param_values=param_values, denominator_of_input=denominator_of_input,
                                                                                splits=splits, train_data=train, n_epochs=n_epochs,
                                                                                folds=k, data_pre=data_pre,
                                                                                transform=transform, runs_dir=runs_dir, model_dir=model_dir)


    # load test dataset
    if train_and_test:
        test_loader = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
    else:
        test_loader = DataLoader(train, batch_size=1, shuffle=False, drop_last=True)

    # use best model to predict using test dataset.
    predictions, values = best_opt.evaluate(
        test_loader,
        model_ori=best_model,
        optimizer_ori=best_optimizer,
        best_model_path=best_model_path,
        batch_size=1,
        n_features=train.tensors[0].shape[1]
    )
    # predictions = scaler_y.inverse_transform(predictions)
    # values = scaler_y.inverse_transform(values)
    result_matrix = ew.calculate_metrics_true_pred(values, predictions)

    # check accuracy measures
    result_matrix
    # save predictions to csv files
    pd.DataFrame(predictions).to_csv(resultCSVPath, index=False, na_rep=0)

if __name__=='__main__':
    main()