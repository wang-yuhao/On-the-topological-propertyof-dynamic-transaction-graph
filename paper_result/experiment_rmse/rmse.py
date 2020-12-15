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
from xgboost.sklearn import XGBClassifier
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



PRICED_BITCOIN_FILE_PATH = "C:/Users/wang.yuhao/Documents/ChainNet/data/original_data/pricedBitcoin2009-2018.csv"
DAILY_OCCURRENCE_FILE_PATH = "C:/Users/wang.yuhao/Documents/ChainNet/data/original_data/dailyOccmatrices/"


ROW = -1
COLUMN = -1
TEST_SPLIT = 0.01



ALL_YEAR_INPUT_ALLOWED = False
YEAR = 2017

# Baseline

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
            occurrence_data = np.concatenate((occurrence_data, daily_occurrence_normalized_matrix), axis=1)
   #print("merge_data shape: {} occurrence_data: {} ".format(occurrence_data.shape, occurrence_data))
    return occurrence_data
    
    
def get_normalized_matrix_from_file(day, year, totaltx):
    daily_occurrence_matrix_path_name = DAILY_OCCURRENCE_FILE_PATH + "occ" + str(year) + '{:03}'.format(day) + '.csv'
    daily_occurence_matrix = pd.read_csv(daily_occurrence_matrix_path_name, sep=",", header=None).values
   #print("daily_occurence_matrix.size: ", daily_occurence_matrix.size, daily_occurence_matrix.shape)
   #print("np.asarray(daily_occurence_matrix): ",np.asarray(daily_occurence_matrix))
   #print("np.asarray(daily_occurence_matrix).reshape(1, daily_occurence_matrix.size): ",np.asarray(daily_occurence_matrix).reshape(1, daily_occurence_matrix.size), np.asarray(daily_occurence_matrix).reshape(1, daily_occurence_matrix.size).shape, np.asarray(daily_occurence_matrix).reshape(1, daily_occurence_matrix.size).size)
   #print("totaltx: ",totaltx)
   #print("np.asarray(daily_occurence_matrix).reshape(1, daily_occurence_matrix.size)/totaltx: ",np.asarray(daily_occurence_matrix).reshape(1, daily_occurence_matrix.size)/totaltx)
    return np.asarray(daily_occurence_matrix).reshape(1, daily_occurence_matrix.size)/totaltx

def get_daily_occurrence_matrices(priced_bitcoin, current_row, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
   #print("priced_bitcoin: ", priced_bitcoin, priced_bitcoin.shape)
   #print("current_row: ", current_row, current_row.shape)
    previous_price_data = np.array([], dtype=np.float32)
    occurrence_data = np.array([], dtype=np.float32)
    for index, row in priced_bitcoin.iterrows():
        if not ((row.values == current_row.values).all()):
            previous_price_data = np.append(previous_price_data, row['price'])
            previous_price_data = np.append(previous_price_data, row['totaltx'])
           #print("previous_price_data: ", previous_price_data,row['day'], row['year'], row['totaltx'])

    
   #print("occurrence_data: ", occurrence_data)
    if(is_price_of_previous_days_allowed):
       #print("previous_price_data: ", np.asarray(previous_price_data).reshape(1, -1), np.asarray(previous_price_data).reshape(1, -1).shape)
        occurrence_data = np.asarray(previous_price_data).reshape(1, -1)
    occurrence_input = np.concatenate((occurrence_data, np.asarray(current_row['price']).reshape(1,1)), axis=1)
   #print("current_row: ", current_row, current_row.shape)
   #print(" price occurrence_input: ", np.asarray(current_row['price']).reshape(1,1), (np.asarray(current_row['price']).reshape(1,1)).shape)
   #print("concatenate with price occurrence_input: ", occurrence_input, occurrence_input.shape)
    occurrence_input = np.concatenate((occurrence_input, np.asarray(current_row['day']).reshape(1,1)), axis=1)
   #print(" price occurrence_input: ", np.asarray(current_row['day']).reshape(1,1), (np.asarray(current_row['day']).reshape(1,1)).shape)

   #print("concatenate with day occurrence_input: ", occurrence_input, occurrence_input.shape)
    return occurrence_input


def rf_base_rmse(train_input, train_target, test_input, test_target):
    rf_regression = RandomForestRegressor(max_depth=2, random_state=0)
    rf_regression.fit(train_input, train_target.ravel() )
    predicted = rf_regression.predict(test_input)
    rf_base_rmse = np.sqrt(metrics.mean_squared_error(test_target, predicted))
    return rf_base_rmse

def gp_base_rmse(train_input, train_target, test_input, test_target):
    param = {
        'kernel': RationalQuadratic(alpha=0.0001, length_scale=1),
        'n_restarts_optimizer': 2
        }
    
    adj_params = {'kernel': [RationalQuadratic(alpha=0.0001, length_scale=1), RationalQuadratic(alpha=0.001, length_scale=1), RationalQuadratic(alpha=0.01,length_scale=1), RationalQuadratic(alpha=0.1, length_scale=1), RationalQuadratic(alpha=1, length_scale=1), RationalQuadratic(alpha=10, length_scale=1),
                             RationalQuadratic(alpha=0.0001, length_scale=1), RationalQuadratic(alpha=0.001, length_scale=1), RationalQuadratic(alpha=0.01,length_scale=1), RationalQuadratic(alpha=0.1, length_scale=1), RationalQuadratic(alpha=1, length_scale=1), RationalQuadratic(alpha=10, length_scale=1)],
                 'n_restarts_optimizer': [2]}
    gpr = GaussianProcessRegressor(**param)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    cscv = GridSearchCV(gpr, adj_params, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    cscv.fit(train_input,train_target)

    #print("cv_results_:",cscv.cv_results_)
    print("best_params_: ",cscv.best_params_)
    gpr = GaussianProcessRegressor(**cscv.best_params_)
    

    gpr.fit(train_input, train_target)
    mu, cov = gpr.predict(test_input, return_cov=True)
    test_y = mu.ravel()
    #uncertainty = 1.96 * np.sqrt(np.diag(cov))
    gp_base_rmse = np.sqrt(metrics.mean_squared_error(test_target, test_y))
    print(gp_base_rmse)
    return gp_base_rmse
    
def enet_base_rmse(train_input, train_target, test_input, test_target):
    param = {
        'alpha': 0.0001,
        'l1_ratio': 0.001,
        }
    elastic = linear_model.ElasticNet(**param)

    adj_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                 'l1_ratio': [0.001, 0.005 ,0.01, 0.05, 0.1, 0.5, 1]}
                 #'max_iter': [100000]}

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    cscv = GridSearchCV(elastic, adj_params, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    cscv.fit(train_input, train_target)

    elastic= linear_model.ElasticNet(**cscv.best_params_)
    elastic.fit(train_input,train_target.ravel())  
    predicted = elastic.predict(test_input) 
    enet_base_rmse = np.sqrt(metrics.mean_squared_error(test_target, predicted))
    print("enet_base_rmse: ", enet_base_rmse)
    #print ("RMSE:", np.sqrt(metrics.mean_squared_error(test_target, predicted))) 
    return enet_base_rmse

def xgbt_base_rmse(train_input, train_target, test_input, test_target):
    param = {
        'n_estimators':10,
        'learning_rate': 0.01,
        }

    adj_params = {
        'n_estimators':[10],
        'learning_rate': [0.01],
    #    'n_estimators':[10,50,100,200,300,400,500,1000],
    #    'learning_rate': [0.01, 0.1, 1] 
    }

    xgbt = XGBClassifier(**param)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    cscv = GridSearchCV(xgbt, adj_params, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    cscv.fit(train_input, train_target)
    print("cv_results_:",cscv.cv_results_)
    print("best_params_: ",cscv.best_params_)
    xgbt= XGBClassifier(**cscv.best_params_)
    xgbt.fit(train_input,train_target.ravel())  
    predicted = xgbt.predict(test_input) 
    xgbt_base_rmse = np.sqrt(metrics.mean_squared_error(test_target, predicted))
    print("xgbt_base_rmse: ", xgbt_base_rmse)
    #print ("RMSE:", np.sqrt(metrics.mean_squared_error(test_target, predicted))) 
    return xgbt_base_rmse

    
def arimax_base_rmse(train_input, train_target, test_input, test_target):
    train_input_diff_arr = np.array([])
    train_columns_name = []
    train_input_column = int(train_input.shape[1])
    for i in range(train_input_column):
        if(i%2==0):
            train_columns_name.append('price_' + str(i))
        else:
            train_columns_name.append('totaltx_' + str(i))
        train_input_diff = np.diff(train_input[:,i] )
        if i == 0:
            train_input_diff_arr = train_input_diff
        else:
            train_input_diff_arr = np.dstack((train_input_diff_arr, train_input_diff))

    columns_name = copy.deepcopy(train_columns_name)
    columns_name.append('current_price')
    train_target_diff = np.diff(train_target )
    train_input_diff_arr = np.dstack((train_input_diff_arr, train_target_diff))


    train_input_diff_arr = pd.DataFrame(train_input_diff_arr[0], columns = columns_name)


    best_column = 'totaltx_' + str(int(train_input_column/2))
    model  = pf.ARIMAX(data=train_input_diff_arr,formula="current_price~" + str(best_column),ar=1,ma=3,integ=0)

    model_1 = model.fit("MLE")
    model_1.summary()


    test_input_pd = pd.DataFrame(test_input, columns = train_columns_name)
    test_target_pd = pd.DataFrame(test_target, columns = ['current_price'])
    test_input_target = pd.concat([test_input_pd, test_target_pd], axis=1)

    pred = model.predict(h=test_input_target.shape[0],
                       oos_data=test_input_target, 
                           intervals=True, )

    arimax_base_rmse = mean_squared_error(test_target,(train_input[99,4])+pred.current_price)
    return arimax_base_rmse

def run_print_model(train_input, train_target, test_input, test_target, train_days, test_days):
    rf_base_rmse = rf_base_rmse(train_input, train_target, test_input, test_target)
    xgbt_base_rmse = xgbt_base_rmse(train_input, train_target, test_input, test_target)
    gp_base_rmse = gp_base_rmse(train_input, train_target, test_input, test_target)
    enet_base_rmse = enet_base_rmse(train_input, train_target, test_input, test_target)
    arimax_base_rmse = arimax_base_rmse(train_input, train_target, test_input, test_target)
    rmse = [rf_base_rmse, xgbt_base_rmse, gp_base_rmse, enet_base_rmse, arimax_base_rmse]
    #print_results(predicted, test_target, original_log_return, predicted_log_return, cost, test_days, rmse)
    #return rf_base_rmse

def preprocess_data(window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
    priced_bitcoin = pd.read_csv(PRICED_BITCOIN_FILE_PATH, sep=",")
    if(ALL_YEAR_INPUT_ALLOWED):
        pass
    else:
        priced_bitcoin = priced_bitcoin[priced_bitcoin['year']==YEAR].reset_index(drop=True)
    
    # get normalized occurence matrix in a flat format and merge with totaltx
    daily_occurrence_input = np.array([],dtype=np.float32)
    temp = np.array([], dtype=np.float32)
    for current_index, current_row in priced_bitcoin.iterrows():
        if(current_index<(window_size+prediction_horizon-1)):
            pass
        else:
            start_index = current_index - (window_size + prediction_horizon) + 1
            end_index = current_index - prediction_horizon
            temp = get_daily_occurrence_matrices(priced_bitcoin[start_index:end_index+1], current_row, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)    
            #print("1st temp: ", temp, temp.shape)
        if(daily_occurrence_input.size == 0):
            daily_occurrence_input = temp
        else:
            #print("daily_occurrence_input: ", daily_occurrence_input, daily_occurrence_input.shape)
            #print("temp: ", temp, temp.shape)
            daily_occurrence_input = np.concatenate((daily_occurrence_input, temp), axis=0)
            #print("return daily_occurrence_input:", daily_occurrence_input, daily_occurrence_input.shape)
            
        #if current_index == 108:
            #print("daily_occurrence_input: ", daily_occurrence_input, daily_occurrence_input.shape)
    return daily_occurrence_input
        
        
    
def initialize_setting(window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
    data = preprocess_data(window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)
    #train, test = train_test_split(data, test_size=TEST_SPLIT)
    #data = pd.DataFrame(data)
    train = data[0:100, :]
    test = data[101, :].reshape(1, -1)
    #print(" train, test shape",train.shape, test.shape)
    #print(" train, test",train, test)
    
    x_train, x_test, train_days, test_days = exclude_days(train, test)
    #print("x_train:", x_train)
    row, column = x_train.shape
    train_target = np.asarray(x_train[:, -1]).reshape(-1)
    train_input = x_train[:, 0:column - 1]
    #x_test = x_test.reshape(-1,1)
    test_target = x_test[: , -1]
    test_input = x_test[ : , 0:column - 1]
    return train_input, train_target, test_input, test_target, train_days, test_days

parameter_dict = {#0: dict({'is_price_of_previous_days_allowed':True, 'aggregation_of_previous_days_allowed':True})}
                  1: dict({'is_price_of_previous_days_allowed':True, 'aggregation_of_previous_days_allowed':False})}

for step in parameter_dict:
    gc.collect()
    evalParameter = parameter_dict.get(step)
    is_price_of_previous_days_allowed = evalParameter.get('is_price_of_previous_days_allowed')
    aggregation_of_previous_days_allowed = evalParameter.get('aggregation_of_previous_days_allowed')
    print("IS_PRICE_OF_PREVIOUS_DAYS_ALLOWED: ", is_price_of_previous_days_allowed)
    print("AGGREGATION_OF_PREVIOUS_DAYS_ALLOWED: ", aggregation_of_previous_days_allowed)
    window_size_array = [3, 5, 7]
    horizon_size_array = [1, 2, 5, 7, 10, 15, 20, 25, 30]
    for window_size in window_size_array:
        print('WINDOW_SIZE: ', window_size)
        rmse_array = []
        for prediction_horizon in horizon_size_array:
            print("PREDICTION_HORIZON: ", prediction_horizon)
            train_input, train_target, test_input, test_target, train_days, test_days = initialize_setting(window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)
            #print("train_input, train_target: ",train_input, train_target, train_input.shape, train_target.shape)
            #print("test_input, test_target",test_input, test_target, test_input.shape, test_target.shape)
            #print("train_days, test_days: ",train_days, test_days)
            rmse = run_print_model(train_input, train_target, test_input, test_target, train_days, test_days)
            rmse_array.append(rmse)
            
        print("rmse_array:", rmse_array)    
        plt.plot(horizon_size_array, rmse_array)
        plt.xlabel('WINDOW = {}'.format(window_size))
        #plt.show()
        
        