# Process 2017 bitcoin data for Informer
import pandas as pd
import os
from sklearn.decomposition import PCA
import datetime
import math 
import pandas as pd
import numpy as np
import torch

BETTI_NUMBER_0_PATH = "/content/drive/MyDrive/bitcoin/CoinWorks/data/betti_0(100).csv"
BETTI_NUMBER_1_PATH = "/content/drive/MyDrive/bitcoin/CoinWorks/data/betti_1(100).csv"

AMOMAT_DIR = "/content/drive/MyDrive/bitcoin/CoinWorks/data/amo/"
OCCMAT_DIR = "/content/drive/MyDrive/bitcoin/CoinWorks/data/occ/"
PRICE_PATH = "/content/drive/MyDrive/bitcoin/CoinWorks/data/pricedBitcoin2009-2018.csv"
PROCESSED_DIR = "/content/drive/MyDrive/aliyun/processed_data/2017/"
#TOTALTX_DIR = "/content/drive/MyDrive/aliyun/bitcoin_totaltx_2018_2020.csv"
#PERIOD = [2018, 2019, 2020]

def getBetweenDay(begin_date, end_date):
    date_list = []
    date_arr = []
    date_unix_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    print("begin_date:",begin_date)
    # end_date = datetime.datetime.strptime(time.strftime('%Y-%m-%d', time.localtime(time.time())), "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    print("end_date:",end_date)
    while begin_date <= end_date:
        date_unix = math.trunc(begin_date.replace(tzinfo=datetime.timezone.utc).timestamp()*1000)
        date_unix_list.append(date_unix)
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        date_arr.append([date_str, date_unix])
        begin_date += datetime.timedelta(days=1)  
    return  np.asarray(date_arr)

def combine_features_with_data(dataset_model):
    data_price = pd.read_csv(PRICE_PATH)[-365:][["totaltx","price"]].reset_index(drop=True)

    #btc_price_2018_2020 = data_price.Open.str.replace(",","")
    #total_tx = pd.read_csv(TOTALTX_DIR, index_col=0)
    date_arr = pd.DataFrame(getBetweenDay("2017-01-01", "2017-12-31"))[0]
    btc_2017 = pd.concat([data_price, date_arr], axis = 1)
    btc_2017.columns = ["totaltx", "price", "date"]
    data_feature = pd.DataFrame([])
    
    if dataset_model == "betti":
        #for YEAR in PERIOD:
          #for file_name in os.listdir(BETTI_NUMBER_DIR):
        feature_betti_0 = pd.read_csv(BETTI_NUMBER_0_PATH, index_col=0).loc[:, "V1":"V50"]
        feature_betti_1 = pd.read_csv(BETTI_NUMBER_1_PATH, index_col=0).loc[:, "V1":"V50"]
        feature_betti_number = pd.concat([feature_betti_0,feature_betti_1], axis = 1)
        data_feature = pd.concat([data_feature,feature_betti_number]).reset_index(drop=True)
        #data_feature.to_csv("data_feature.csv")
        print("data_feature:",data_feature)
    elif dataset_model == "betti_der":
        feature_betti_0 = pd.read_csv(BETTI_NUMBER_0_PATH, index_col=0).loc[:, "V1":"V50"]
        feature_betti_1 = pd.read_csv(BETTI_NUMBER_1_PATH, index_col=0).loc[:, "V1":"V50"]
        feature_betti_0_der = pd.read_csv(BETTI_NUMBER_0_PATH, index_col=0).diff(axis=1)
        feature_betti_1_der = pd.read_csv(BETTI_NUMBER_1_PATH, index_col=0).diff(axis=1)
        feature_betti_0_der_50 = feature_betti_0_der.loc[:, "V2":"V51"]
        feature_betti_1_der_50 = feature_betti_1_der.loc[:, "V2":"V51"]
        feature_betti_total = pd.concat([feature_betti_0, feature_betti_1, feature_betti_0_der_50, feature_betti_1_der_50], axis=1)
        data_feature = pd.concat([data_feature,feature_betti_total]).reset_index(drop=True)

    elif dataset_model == "fl":
        for day in range(1,366):
            feature = pd.read_csv(OCCMAT_DIR + "occ2017" + '{:03}'.format(day) + '.csv', header=None, index_col=False).to_numpy()
            feature = pd.DataFrame(feature.flatten()).T
            data_feature = pd.concat([data_feature,feature], axis = 0)

    data_feature.to_csv(PROCESSED_DIR + dataset_model+"_orig.csv")
    print("data_feature:",data_feature)
    if len(data_feature) > 0:
        pca = PCA(n_components = 20)
        pca.fit(data_feature)
        data_feature = pd.DataFrame(pca.transform(data_feature))
    print("pca data_feature:",data_feature)

    data_combined = pd.concat([btc_2017,data_feature], axis=1)
    cols = data_combined.columns.tolist()
    cols = cols[2:] + cols[:2]
    data_combined = data_combined[cols] 
    data_combined.to_csv(PROCESSED_DIR + dataset_model+".csv", index=False)
    print(data_combined)

for dataset_model in ["base", "betti","betti_der", "fl"]:
    combine_features_with_data(dataset_model)
