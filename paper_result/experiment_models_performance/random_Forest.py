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
from sklearn import metrics
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



PRICED_BITCOIN_FILE_PATH = "C:/Users/wang.yuhao/Documents/ChainNet/data/original_data/pricedBitcoin2009-2018.csv"
DAILY_OCCURRENCE_FILE_PATH = "C:/Users/wang.yuhao/Documents/ChainNet/data/original_data/dailyOccmatrices/"


ROW = -1
COLUMN = -1
TEST_SPLIT = 0.01



ALL_YEAR_INPUT_ALLOWED = False
YEAR = 2017


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


def print_results(predicted, test_target, original_log_return, predicted_log_return, cost, test_days, rmse):
    myFile = open('C:/Users/wang.yuhao/Documents/ChainNet/data/processed_data/rmse/' + str(YEAR) + ".csv", 'a')
    if(window_size == 1):
        myFile.write('IS_PRICE_OF_PREVIOUS_DAYS_ALLOWED:' + str(is_price_of_previous_days_allowed) + '\n')
        myFile.write('AGGREGATION_OF_PREVIOUS_DAYS_ALLOWED:' + str(aggregation_of_previous_days_allowed) + '\n')

        myFile.write('PREDICTION_HORIZON:' + str(prediction_horizon) + '\n')
    myFile.write('WINDOW_SIZE:' + str(window_size) + '\n')

    for p, t, o_l, p_l, t_d in zip(predicted, test_target, original_log_return, predicted_log_return, test_days):
        myFile.write(str(p) + "\t" + str(t) + "\t" + str(o_l) + "\t" + str(p_l) + "\t" + str(t_d) + '\n')
    myFile.write('TEST_COST:' + str(cost) + '\n')
    myFile.write('RMSE:' + str(rmse) + '\n')
    myFile.close()


def run_print_model(train_input, train_target, test_input, test_target, train_days, test_days):
    rf_regression = RandomForestRegressor(max_depth=2, random_state=0)
    rf_regression.fit(train_input, train_target.ravel() )
    predicted = rf_regression.predict(test_input)
    original_log_return = np.log(np.asarray(test_target).reshape(-1,)/test_input[:,-1])
    predicted_log_return = np.log(np.asarray(predicted).reshape(-1,)/test_input[:,-1])
    cost = sum(np.absolute(original_log_return-predicted_log_return))/original_log_return.size
    rmse = np.sqrt(metrics.mean_squared_error(test_target, predicted))
    print_results(predicted, test_target, original_log_return, predicted_log_return, cost, test_days, rmse)
    return rmse

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
    horizon_size_array = range(1,31)
    for window_size in window_size_array:
        print('WINDOW_SIZE: ', window_size)
        rmse_array = []
        for prediction_horizon in range(1, 31):
            print("PREDICTION_HORIZON: ", prediction_horizon)
            train_input, train_target, test_input, test_target, train_days, test_days = initialize_setting(window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)
            #print("train_input, train_target: ",train_input, train_target, train_input.shape, train_target.shape)
            #print("test_input, test_target",test_input, test_target, test_input.shape, test_target.shape)
            #print("train_days, test_days: ",train_days, test_days)
            rmse = run_print_model(train_input, train_target, test_input, test_target, train_days, test_days)
            rmse_array.append(rmse)
        plt.plot(rmse_array)
        plt.xlabel('WINDOW = {}'.format(window_size))
        plt.show()
