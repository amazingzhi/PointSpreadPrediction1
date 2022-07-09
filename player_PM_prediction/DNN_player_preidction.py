# import libraries
import torch
import pandas as pd
from torch.utils.data import DataLoader
import example_web as ew
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

""" 
ToDo YOU HAVE TO CREATE 'checkpoint' and 'model' and 'predictions' folders in your working directory
Todo: after you change datasets, put "tensorboard --logdir player_PM_prediction/runs_original ; runs_feature_selected ; runs_PCA" into terminal and
    change working directory. after you have run the program, put "tensorboard --logdir player_PM_prediction/runs_original ; runs_feature_selected ; runs_PCA" into terminal again to see your results in tensorboard. 
todo : change variables with comments after codes when you change datasets"""
# global parameters set up
    # training parameters
n_epochs = 30
k = 3
shuffle = True
random_state = 42
splits = KFold(n_splits=k,shuffle=shuffle,random_state=random_state)
    # model parameters
learning_rates=[1e-4]  # , 1e-3, 1e-2
batch_sizes=[64]  # 16, 32
weight_decays=[1e-4]  # 1e-6, , 1e-2
    # tensorboard
tb = SummaryWriter()
    #read data parameters
train_and_test = False  # True
data_pre = 'original'  # 'feature_selected', 'PCA'
train_path = 'player_prediction_data_feature_selection_and_seconds.csv'
#'data/games and player from 2004 to 2020/player_prediction_data/player_prediction_data_feature_selection_and_seconds.csv'
test_path = ''
target_col = 'PLUS_MINUS'
col_to_drop = ['GAME_ID', 'PLAYER_ID', 'TEAM_ID', 'OPPOID', 'GAME_DATE_EST', 'PLUS_MINUS']
transform = 'standard'  # 'standard', "maxabs", "robust".
scaler = ew.get_scaler(transform)
resultCSVPath = f'player_PM_prediction/predictions/{data_pre}_DNN_{transform}.csv'

## see if cuda available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

def main():
    if train_and_test:
        train, Train_X, test = ew.read_data(train_and_test=train_and_test, train_path=train_path,
                                            test_path=test_path,
                                            target_col=target_col,col_to_drop=col_to_drop,
                                            scaler=scaler)
    else:
        train, Train_X = ew.read_data(train_and_test=train_and_test, train_path=train_path, test_path=test_path,
                                      target_col=target_col,col_to_drop=col_to_drop,scaler=scaler)
    # model parameters
    model_params, input_dim = ew.model_parameters(len(Train_X.columns))

    # training parameters
    param_values = ew.training_parameters(learning_rates=learning_rates,batch_sizes=batch_sizes,
                                          weight_decays=weight_decays,model_params=model_params)

    # training with cross validation
    best_opt, best_model, best_optimizer, best_model_path = ew.cross_validation(param_values=param_values, model_params=model_params,
                                                                                splits=splits, train_data=train, n_epochs=n_epochs,
                                                                                input_dim=input_dim, folds=k, data_pre=data_pre,
                                                                                transform=transform)


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
        n_features=input_dim
    )
    predictions = scaler.inverse_transform(predictions)
    values = scaler.inverse_transform(values)
    result_matrix = ew.calculate_metrics_true_pred(values, predictions)

    # check accuracy measures
    result_matrix
    # save predictions to csv files
    pd.DataFrame(predictions).to_csv(resultCSVPath, index=False, na_rep=0)

if __name__=='__main__':
    main()