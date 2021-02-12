from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np, tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import os
import csv
import gc
from sklearn.metrics import mean_squared_error
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn import linear_model
from xgboost.sklearn import XGBRegressor
from sklearn.decomposition import PCA
import copy
import pyflux as pf
import datetime
from pprint import pprint
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PRICED_BITCOIN_FILE_PATH = "C:/Users/wang.yuhao/Documents/ChainNet/data/original_data/pricedBitcoin2009-2018.csv"
betti0_input_path = "C:/Users/wang.yuhao/Documents/ChainNet/data/original_data/betti_0(100).csv"
betti1_input_path = "C:/Users/wang.yuhao/Documents/ChainNet/data/original_data/betti_1(100).csv"
DAILY_FILTERED_OCCURRENCE_FILE_PATH = "C:/Users/wang.yuhao/Documents/ChainNet/data/original_data/filteredDailyOccMatrices/"
DAILY_OCCURRENCE_FILE_PATH = "C:/Users/wang.yuhao/Documents/CoinWorks-master/data/dailyVrAmoMatrices/dailyOccmatrices/"


ROW = -1
COLUMN = -1
TEST_SPLIT = 0.01


ALL_YEAR_INPUT_ALLOWED = False

SLIDING_BATCH_SIZE = 200
TEST_LENGTH = 100
YEAR = 2017


from sklearn.metrics import mean_squared_error
from sklearn import metrics
import matplotlib.pyplot as plt

def exclude_days(train, test):

    row, column = train.shape
    train_days = np.asarray(train[:, -1]).reshape(-1, 1)
    x_train = train[:, 0:column - 1]
    test_days = np.asarray(test[:, -1]).reshape(-1, 1)
    x_test = test[:, 0:column - 1]

    return x_train, x_test, train_days, test_days

def merge_data(occurrence_data, daily_occurrence_normalized_matrix, aggregation_of_previous_days_allowed):
    if(aggregation_of_previous_days_allowed):
        if(occurrence_data.size==0):
            occurrence_data = daily_occurrence_normalized_matrix
        else:
            occurrence_data = np.add(occurrence_data, daily_occurrence_normalized_matrix)
    else:
        if(occurrence_data.size == 0):
            occurrence_data = daily_occurrence_normalized_matrix
        else:
            occurrence_data = np.concatenate((occurrence_data, daily_occurrence_normalized_matrix), axis=0)

    return occurrence_data
    

def baseline(priced_bitcoin, current_row, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
    previous_price_data = np.array([], dtype=np.float32)
    occurrence_data = np.array([], dtype=np.float32)
    for index, row in priced_bitcoin.iterrows():
        if not ((row.values == current_row.values).all()):
            previous_price_data = np.append(previous_price_data, row['price'])

    if(is_price_of_previous_days_allowed):
        occurrence_data = np.asarray(previous_price_data).reshape(1, -1)
    occurrence_input = np.concatenate((occurrence_data, np.asarray(current_row['price']).reshape(1,1)), axis=1)

    occurrence_input = np.concatenate((occurrence_input, np.asarray(current_row['day']).reshape(1,1)), axis=1)

    return occurrence_input

def model_1(priced_bitcoin, current_row, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
    previous_price_data = np.array([], dtype=np.float32)
    previous_tx_data = np.array([], dtype=np.float32)
    occurrence_data = np.array([], dtype=np.float32)
    for index, row in priced_bitcoin.iterrows():
        if not ((row.values == current_row.values).all()):
            previous_price_data = np.append(previous_price_data, row['price'])
            previous_tx_data = np.append(previous_tx_data, row['totaltx'])

    previous_price_data = np.append(previous_price_data, previous_tx_data)
    if(is_price_of_previous_days_allowed):
        occurrence_data = np.asarray(previous_price_data).reshape(1, -1)
    occurrence_input = np.concatenate((occurrence_data, np.asarray(current_row['price']).reshape(1,1)), axis=1)
    occurrence_input = np.concatenate((occurrence_input, np.asarray(current_row['day']).reshape(1,1)), axis=1)

    return occurrence_input

def get_split_matrix_from_file(day, year):
    daily_occurence_matrix = np.asarray([],dtype=np.float32)
    daily_occurrence_matrix_path_name = DAILY_OCCURRENCE_FILE_PATH + 'occ'+ str(year) + '{:03}'.format(day) +'.csv'
    daily_occurence_matrix_read = pd.read_csv(daily_occurrence_matrix_path_name, sep=",", header=None).values
    daily_occurence_matrix = daily_occurence_matrix_read

    daily_split_matrix = daily_occurence_matrix[np.triu_indices(8, 1)]

    return daily_split_matrix

def get_cluster_matrix_from_file(day, year):
    daily_occurence_matrix = np.asarray([],dtype=np.float32)
    daily_occurrence_matrix_path_name = DAILY_OCCURRENCE_FILE_PATH + 'occ'+ str(year) + '{:03}'.format(day) +'.csv'
    daily_occurence_matrix_read = pd.read_csv(daily_occurrence_matrix_path_name, sep=",", header=None).values
    
    daily_occurence_matrix_read[:,7] = daily_occurence_matrix_read[:,7:].sum(axis=1)
    daily_occurence_matrix_read_1 = daily_occurence_matrix_read[:,0:8]
    daily_occurence_matrix_read_1[7,:] = daily_occurence_matrix_read_1[7:,:].sum(axis=0)
    daily_occurence_matrix = daily_occurence_matrix_read_1[0:8,:]
    daily_occurence_matrix

    return np.asarray(daily_occurence_matrix).reshape(1, daily_occurence_matrix.size)  

def get_specific_occurrence_from_file(day, year, input_number, output_number):
    daily_specific_occurrence = np.asarray([],dtype=np.float32)
    daily_occurrence_matrix_path_name = DAILY_OCCURRENCE_FILE_PATH + 'occ'+ str(year) + '{:03}'.format(day) +'.csv'
    daily_occurence_matrix_read = pd.read_csv(daily_occurrence_matrix_path_name, sep=",", header=None).values
    daily_specific_occurrence = daily_occurence_matrix_read[input_number - 1, output_number - 1]

    return daily_specific_occurrence


def model_2(priced_bitcoin, current_row, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
    previous_price_data = np.array([], dtype=np.float32)
    cluster_data = np.array([], dtype=np.float32)
    split_data = np.array([], dtype=np.float32)
    for index, row in priced_bitcoin.iterrows():
        if not ((row.values == current_row.values).all()):
            previous_price_data = np.append(previous_price_data, row['price'])
            daily_occurrence_split_matrix = get_split_matrix_from_file(row['day'], row['year'])
            split_data = np.append(split_data, daily_occurrence_split_matrix)
            daily_occurrence_cluster_matrix = get_cluster_matrix_from_file(row['day'], row['year'])
            cluster_data = merge_data(cluster_data, daily_occurrence_cluster_matrix, aggregation_of_previous_days_allowed)

    if (is_price_of_previous_days_allowed):
        cluster_data = np.concatenate((np.asarray(previous_price_data).flatten(), split_data, cluster_data.flatten()),  axis=0)

    occurrence_input = np.concatenate((cluster_data.reshape(1,-1), np.asarray(current_row['price']).reshape(1,1)), axis=1)
    occurrence_input = np.concatenate((occurrence_input, np.asarray(current_row['day']).reshape(1,1)), axis=1)
    return occurrence_input

def model_3(priced_bitcoin, current_row, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
    previous_price_data = np.array([], dtype=np.float32)
    specific_occurrence_data = np.array([], dtype=np.float32)
    for index, row in priced_bitcoin.iterrows():
        if not ((row.values == current_row.values).all()):
            previous_price_data = np.append(previous_price_data, row['price'])
            daily_specific_occurrence = get_specific_occurrence_from_file(row['day'], row['year'], 1, 7)
            specific_occurrence_data = np.append(specific_occurrence_data, daily_specific_occurrence)

    if (is_price_of_previous_days_allowed):
        occurrence_data = np.concatenate((np.asarray(previous_price_data).flatten(), specific_occurrence_data),  axis=0)

    occurrence_input = np.concatenate((occurrence_data.reshape(1,-1), np.asarray(current_row['price']).reshape(1,1)), axis=1)
    occurrence_input = np.concatenate((occurrence_input, np.asarray(current_row['day']).reshape(1,1)), axis=1)
    return occurrence_input

def model_4(priced_bitcoin, current_row, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
    previous_price_data = np.array([], dtype=np.float32)
    specific_occurrence_data_1_7 = np.array([], dtype=np.float32)
    specific_occurrence_data_6_1 = np.array([], dtype=np.float32)
    for index, row in priced_bitcoin.iterrows():
        if not ((row.values == current_row.values).all()):
            previous_price_data = np.append(previous_price_data, row['price'])
            daily_specific_occurrence = get_specific_occurrence_from_file(row['day'], row['year'], 1, 7)
            specific_occurrence_data_1_7 = np.append(specific_occurrence_data_1_7, daily_specific_occurrence)
            daily_specific_occurrence = get_specific_occurrence_from_file(row['day'], row['year'], 6, 1)
            specific_occurrence_data_6_1 = np.append(specific_occurrence_data_6_1, daily_specific_occurrence)

    if (is_price_of_previous_days_allowed):
        occurrence_data = np.concatenate((np.asarray(previous_price_data).flatten(), specific_occurrence_data_1_7, specific_occurrence_data_6_1),  axis=0)

    occurrence_input = np.concatenate((occurrence_data.reshape(1,-1), np.asarray(current_row['price']).reshape(1,1)), axis=1)
    occurrence_input = np.concatenate((occurrence_input, np.asarray(current_row['day']).reshape(1,1)), axis=1)
    return occurrence_input

def model_5(priced_bitcoin, current_row, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
    previous_price_data = np.array([], dtype=np.float32)
    specific_occurrence_data_1_7 = np.array([], dtype=np.float32)
    specific_occurrence_data_6_1 = np.array([], dtype=np.float32)
    specific_occurrence_data_3_3 = np.array([], dtype=np.float32)
    for index, row in priced_bitcoin.iterrows():
        if not ((row.values == current_row.values).all()):
            previous_price_data = np.append(previous_price_data, row['price'])
            daily_specific_occurrence = get_specific_occurrence_from_file(row['day'], row['year'], 1, 7)
            specific_occurrence_data_1_7 = np.append(specific_occurrence_data_1_7, daily_specific_occurrence)
            daily_specific_occurrence = get_specific_occurrence_from_file(row['day'], row['year'], 6, 1)
            specific_occurrence_data_6_1 = np.append(specific_occurrence_data_6_1, daily_specific_occurrence)
            daily_specific_occurrence = get_specific_occurrence_from_file(row['day'], row['year'], 3, 3)
            specific_occurrence_data_3_3 = np.append(specific_occurrence_data_3_3, daily_specific_occurrence)

    if (is_price_of_previous_days_allowed):
        occurrence_data = np.concatenate((np.asarray(previous_price_data).flatten(), specific_occurrence_data_1_7, specific_occurrence_data_6_1, specific_occurrence_data_3_3),  axis=0)

    occurrence_input = np.concatenate((occurrence_data.reshape(1,-1), np.asarray(current_row['price']).reshape(1,1)), axis=1)
    occurrence_input = np.concatenate((occurrence_input, np.asarray(current_row['day']).reshape(1,1)), axis=1)
    return occurrence_input

def read_betti(file_path, day):
    day = day - 1
    betti = pd.read_csv(file_path, index_col=0)
    try:
        betti_50 = betti.iloc[day, 0:50]
    except:
        print("day:",  day)
            
    return betti_50


def rf_mode(train_input, train_target, test_input, test_target):
    param = {
        'n_estimators':500
        }

    rf_regression = RandomForestRegressor(**param)
    rf_regression.fit(train_input, train_target.ravel() )
    rf_predicted = rf_regression.predict(test_input)

    return rf_predicted

def filter_data(priced_bitcoin):
    end_day_of_previous_year = max(priced_bitcoin[priced_bitcoin['year'] == YEAR-1]["day"].values)
    start_index_of_previous_year = end_day_of_previous_year - SLIDING_BATCH_SIZE - window_size
    previous_year_batch = priced_bitcoin[(priced_bitcoin['year'] == YEAR-1) & (priced_bitcoin['day'] > start_index_of_previous_year)]
    input_batch = priced_bitcoin[(priced_bitcoin['year'] >= YEAR) & (priced_bitcoin['year'] <= YEAR)]
    filtered_data = previous_year_batch.append(input_batch)
    filtered_data.insert(0, 'index', range(0, len(filtered_data)))
    filtered_data = filtered_data.reset_index(drop=True)
    return filtered_data


def preprocess_data(dataset_model, window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
    priced_bitcoin = pd.read_csv(PRICED_BITCOIN_FILE_PATH, sep=",")
    if(ALL_YEAR_INPUT_ALLOWED):
        pass
    else:
        
        priced_bitcoin = filter_data(priced_bitcoin)
        
    daily_occurrence_input = np.array([],dtype=np.float32)
    temp = np.array([], dtype=np.float32)
    for current_index, current_row in priced_bitcoin.iterrows():
        if(current_index<(window_size+prediction_horizon-1)):
            pass
        else:
            start_index = current_index - (window_size + prediction_horizon) + 1
            end_index = current_index - prediction_horizon
            if(dataset_model=="baseline"):
                temp = baseline(priced_bitcoin[start_index:end_index+1], current_row, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)    
            elif(dataset_model=="model_1"):
                temp = model_1(priced_bitcoin[start_index:end_index+1], current_row, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)    
            elif(dataset_model=="model_2"):
                temp = model_2(priced_bitcoin[start_index:end_index+1], current_row, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)    
            elif(dataset_model=="model_3"):
                temp = model_3(priced_bitcoin[start_index:end_index+1], current_row, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)  
            elif(dataset_model=="model_4"):
                temp = model_4(priced_bitcoin[start_index:end_index+1], current_row, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed) 
            elif(dataset_model=="model_5"):
                temp = model_5(priced_bitcoin[start_index:end_index+1], current_row, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)   
            else:
                sys.exit("Dataset model support only baseline, betti, fl and betti_der!")
        if(daily_occurrence_input.size == 0):
            daily_occurrence_input = temp
        else:
            daily_occurrence_input = np.concatenate((daily_occurrence_input, temp), axis=0)
    return daily_occurrence_input

    
   
def initialize_setting( features, price, day, test_start, dataset_model, window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
    train_target = price[test_start : test_start + 200]
    train_days = day[test_start : test_start + 200]
    pca_features = features[test_start : test_start + 200+1, :]

    train_input = pca_features[0 : 200, :]
    test_target = price[test_start + 200]
    test_days = day[test_start + 200]
    test_input = pca_features[ 200, :].reshape(1, -1)
    return train_input, train_target, test_input, test_target, train_days, test_days

def run_print_model(train_input, train_target, test_input, test_target, train_days, test_days):

    rf_prediction = rf_mode(train_input, train_target, test_input, test_target)
    xgbt_prediction = xgbt_mode(train_input, train_target, test_input, test_target)
    gp_prediction = gp_mode(train_input, train_target, test_input, test_target)
    enet_prediction = enet_mode(train_input, train_target, test_input, test_target)
    
    return rf_prediction, xgbt_prediction, gp_prediction, enet_prediction

def rmse_comparison(rmse_path):
    rmse_baseline = pd.read_csv(rmse_path + "rmse_baseline_total.csv", sep=",", index_col=0)
    rmse_model_1 = pd.read_csv(rmse_path + "rmse_model_1_total.csv", sep=",", index_col=0)
    rmse_model_2 = pd.read_csv(rmse_path + "rmse_model_2_total.csv", sep=",", index_col=0)
    rmse_model_3 = pd.read_csv(rmse_path + "rmse_model_3_total.csv", sep=",", index_col=0)
    rmse_model_4 = pd.read_csv(rmse_path + "rmse_model_4_total.csv", sep=",", index_col=0)
    rmse_model_5 = pd.read_csv(rmse_path + "rmse_model_5_total.csv", sep=",", index_col=0)

    rmse_baseline_arr = []
    rmse_model_1_arr = []
    rmse_model_2_arr = []
    rmse_model_3_arr = []
    rmse_model_4_arr = []
    rmse_model_5_arr = []

    for i in range(9):
        print(rmse_model_1.iloc[i,i])
        rmse_model_1_arr.append(-(rmse_model_1.iloc[i,i]-rmse_baseline.iloc[i,i])/rmse_baseline.iloc[i,i]*100)
        rmse_model_2_arr.append(-(rmse_model_2.iloc[i,i]-rmse_baseline.iloc[i,i])/rmse_baseline.iloc[i,i]*100)
        rmse_model_3_arr.append(-(rmse_model_3.iloc[i,i]-rmse_baseline.iloc[i,i])/rmse_baseline.iloc[i,i]*100)
        rmse_model_4_arr.append(-(rmse_model_4.iloc[i,i]-rmse_baseline.iloc[i,i])/rmse_baseline.iloc[i,i]*100)
        rmse_model_5_arr.append(-(rmse_model_5.iloc[i,i]-rmse_baseline.iloc[i,i])/rmse_baseline.iloc[i,i]*100)

    rmse_models = np.vstack([rmse_model_1_arr, rmse_model_2_arr, rmse_model_3_arr, rmse_model_4_arr, rmse_model_5_arr])
    rmse_models_df = pd.DataFrame(rmse_models.T, columns = ['model 1' , 'model 2', 'model 3', 'model 4', 'model 5'], index=[1,2,5,7,10,15,20,25,30]) 

    pred_fig = rmse_models_df.plot(figsize=(10,6),title="% Change (decrease) in RMSE compared to the baseline model.")
    pred_fig.set_xlabel("Prediction horizon (h)")
    pred_fig.set_ylabel("% Decrease in RMSE, compared to Baseline model")
    pred_fig.set_yticks([0,5,10,15,20])
    pred_fig.set_ylim(-5,25)
    pred_fig.figure.savefig(pred_path + "rmse_comparison_total_fig.png")


def plot_horizon(pred_path, horizon_size):
    priced_bitcoin = pd.read_csv(PRICED_BITCOIN_FILE_PATH, sep=",")
    observed = priced_bitcoin[priced_bitcoin["year"]==2016]["price"].reset_index(drop=True)[1:330]
    pred_M0 = pd.read_csv(pred_path + "pred_baseline_3_30.csv", sep=",", header=0, index_col=0)
    pred_M1 = pd.read_csv(pred_path + "pred_model_1_3_30.csv", sep=",", header=0, index_col=0)
    pred_M5 = pd.read_csv(pred_path + "pred_model_5_3_30.csv", sep=",", header=0, index_col=0)
    pred_comparison_1 =  pd.concat([pred_M0["rf_baseline_prediction_" + str(horizon_size)], pred_M1["rf_model_1_prediction_" + str(horizon_size)], pred_M5["rf_model_5_prediction_" + str(horizon_size)], observed], axis=1)
    pred_comparison_1.columns = ["Model 0", "Model 1", "Model 5", "Observed"]
    pred_fig = pred_comparison_1.plot(figsize=(15,9), color={'blue','gold', 'green', 'black'})
    pred_fig.figure.savefig(pred_path + "pred_of_horizon_"+str(horizon_size))

def split_process(data,dataset_model,window_size):
    if dataset_model == "model_1":
        baseline_features = data[: , 0:-2]
    else:
        baseline_features = data[:, 0:-2]
    fl_features = data[: , window_size:-2]
    price = data[:, -2]
    day = data[:,-1]

    return baseline_features, fl_features, price, day

parameter_dict = {#0: dict({'is_price_of_previous_days_allowed':True, 'aggregation_of_previous_days_allowed':True})}
                  1: dict({'is_price_of_previous_days_allowed':True, 'aggregation_of_previous_days_allowed':False})}

for step in parameter_dict:
    start_time = time.time()
    t = datetime.datetime.now()
    dir_name = t.strftime('%m_%d___%H_%M')
    drive_path = "chainlet_processed_data/"+dir_name
    if not os.path.exists(dir_name):
        os.makedirs(drive_path)
        print("drive_path: ", drive_path)
    result_path = drive_path + "/"
    
    names = locals()
    gc.collect()
    evalParameter = parameter_dict.get(step)
    is_price_of_previous_days_allowed = evalParameter.get('is_price_of_previous_days_allowed')
    aggregation_of_previous_days_allowed = evalParameter.get('aggregation_of_previous_days_allowed')
    print("IS_PRICE_OF_PREVIOUS_DAYS_ALLOWED: ", is_price_of_previous_days_allowed)
    print("AGGREGATION_OF_PREVIOUS_DAYS_ALLOWED: ", aggregation_of_previous_days_allowed)
    window_size_array = [3]
    horizon_size_array = [1, 2, 5, 7, 10, 15, 20, 25, 30]
    dataset_model_array = ["baseline","model_1","model_2", "model_3","model_4","model_5"]
    for dataset_model in dataset_model_array:
        print('dataset_model: ', dataset_model)

        for window_size in window_size_array:
            print('WINDOW_SIZE: ', window_size)
            for prediction_horizon in horizon_size_array:
                data = preprocess_data(dataset_model, window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)
                baseline_features, fl_features, price, day = split_process(data, dataset_model, window_size)
                features = baseline_features
                print("PREDICTION_HORIZON: ", prediction_horizon)
                for test_start in range(1, TEST_LENGTH):
                    train_input, train_target, test_input, test_target, train_days, test_days = initialize_setting( features, price, day, test_start, dataset_model, window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)
                    rf_prediction = rf_mode(train_input, train_target, test_input, test_target)[0]

                    prediction = pd.DataFrame({'rf_' + dataset_model + '_prediction_'+str(prediction_horizon): [rf_prediction]})
                    test_target_df = pd.DataFrame({'test_target': [test_target]})

                    if(test_start==1):
                        prediction_total = prediction
                        test_target_total = test_target_df
                    else:
                        prediction_total = [prediction_total, prediction]
                        test_target_total = [test_target_total, test_target_df]
                        prediction_total = pd.concat(prediction_total)
                        test_target_total = pd.concat(test_target_total)

                print("+++"*10)
                print("prediction_total:",prediction_total)
                print("test_target_total:",test_target_total)
                rmse = ((((prediction_total.sub(test_target_total.values))**2).mean())**0.5).to_frame().T
                print("rmse: ",rmse)

                if(prediction_horizon==1):
                    rmse_total = rmse
                    prediction_total_ = prediction_total
                else:
                    rmse_total = [rmse_total, rmse]
                    rmse_total = pd.concat(rmse_total)
                    print(prediction_total_)
                    print(prediction_total)
                    prediction_total_ = pd.concat([prediction_total_, prediction_total],axis=1)
                    
        names['rmse_' + dataset_model + '_total'] = rmse_total
        names['rmse_' + dataset_model + '_total'].index = pd.Series(horizon_size_array)
        names.get('rmse_' + dataset_model + '_total').to_csv(result_path + "rmse_" + dataset_model + "_total.csv", index=True)
        prediction_total_.index = pd.Series(list(range(1,TEST_LENGTH)))
        prediction_total_.to_csv(result_path + "pred_" + dataset_model + "_"+ str(window_size) +"_"+ str(prediction_horizon) + ".csv", index=True)
        print('rmse_{}_total = {}'.format(dataset_model, names.get('rmse_' + dataset_model + '_total')))  
        print("\n")
        print("--- %s seconds ---" % (time.time() - start_time))       
        print("\n")
    
    rmse_comparison(result_path) 
    # horizon_size_arr = [1, 5, 10, 20]
    # for horizon_size in horizon_size_arr:
    #         plot_horizon(result_path, horizon_size)