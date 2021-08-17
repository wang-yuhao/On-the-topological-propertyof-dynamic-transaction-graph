# Merge 2020 data for Informer
# This process will generate merged base, betti, betti_der, and fl files in PRICESSED_DIR.

import pandas as pd
import os
from sklearn.decomposition import PCA
import datetime
import math 
import pandas as pd
import numpy as np
import torch

BETTI_NUMBER_DIR = "/content/drive/MyDrive/aliyun/betti_number/"
AMOMAT_DIR = "/content/drive/MyDrive/aliyun/amoMat/"
OCCMAT_DIR = "/content/drive/MyDrive/aliyun/occMat/"
PRICE_PATH = "/content/drive/MyDrive/aliyun/bitcoin_2018_2020.csv"
PROCESSED_DIR = "/content/drive/MyDrive/aliyun/processed_data/2020/"
TOTALTX_DIR = "/content/drive/MyDrive/aliyun/bitcoin_totaltx_2018_2020.csv"
PERIOD = [2020]
START_DATE = "2020-01-01"
END_DATE = "2020-12-31"
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
    data_price = pd.read_csv(PRICE_PATH)
    btc_price_2018_2020 = data_price.Open.str.replace(",","")[-366:].reset_index(drop=True)
    total_tx = pd.read_csv(TOTALTX_DIR, index_col=0)[-366:].reset_index(drop=True)
    date_arr = pd.DataFrame(getBetweenDay(START_DATE, END_DATE))[0]
    btc_2018_2020 = pd.concat([total_tx, btc_price_2018_2020, date_arr], axis = 1)
    btc_2018_2020.columns = ["totaltx", "price", "date"]
    print("btc_2018_2020:",btc_2018_2020)
    data_feature = pd.DataFrame([])
    
    if dataset_model == "betti":
      for YEAR in PERIOD:
        #for file_name in os.listdir(BETTI_NUMBER_DIR):
            feature_betti_0 = pd.read_csv(BETTI_NUMBER_DIR + str(YEAR) + "_betti_0.csv", index_col=0).loc[:, "0":"49"]
            feature_betti_1 = pd.read_csv(BETTI_NUMBER_DIR + str(YEAR) + "_betti_1.csv", index_col=0).loc[:, "0":"49"]
            feature_betti_number = pd.concat([feature_betti_0,feature_betti_1], axis = 1)
            data_feature = pd.concat([data_feature,feature_betti_number]).reset_index(drop=True)
      data_feature.to_csv("data_feature.csv")
      print("data_feature:",data_feature)
    elif dataset_model == "betti_der":
        for YEAR in PERIOD:
            feature_betti_0 = pd.read_csv(BETTI_NUMBER_DIR + str(YEAR) + "_betti_0.csv", index_col=0).loc[:, "0":"49"]
            feature_betti_1 = pd.read_csv(BETTI_NUMBER_DIR + str(YEAR) + "_betti_1.csv", index_col=0).loc[:, "0":"49"]
            feature_betti_0_der = pd.read_csv(BETTI_NUMBER_DIR + str(YEAR) + "_betti_0.csv", index_col=0).diff(axis=1)
            feature_betti_1_der = pd.read_csv(BETTI_NUMBER_DIR + str(YEAR) + "_betti_1.csv", index_col=0).diff(axis=1)
            feature_betti_0_der_50 = feature_betti_0_der.loc[:, "1":"50"]
            feature_betti_1_der_50 = feature_betti_1_der.loc[:, "1":"50"]
            feature_betti_total = pd.concat([feature_betti_0, feature_betti_1, feature_betti_0_der_50, feature_betti_1_der_50], axis=1)
            data_feature = pd.concat([data_feature,feature_betti_total]).reset_index(drop=True)

    elif dataset_model == "fl":
        for year in PERIOD:
            for day in getBetweenDay(str(year) + "-01-01", str(year) + "-12-31"):
                feature = pd.read_csv(OCCMAT_DIR + str(year) + "/occ" + day[0] + '.csv', index_col=0).to_numpy()
                feature = pd.DataFrame(feature.flatten()).T
                data_feature = pd.concat([data_feature,feature], axis = 0)

    data_feature.to_csv(PROCESSED_DIR + dataset_model+"_orig.csv")
    print("data_feature:",data_feature)
    if len(data_feature) > 0:
        pca = PCA(n_components = 20)
        pca.fit(data_feature)
        data_feature = pd.DataFrame(pca.transform(data_feature))
    print("pca data_feature:",data_feature)

    data_combined = pd.concat([btc_2018_2020,data_feature], axis=1)
    cols = data_combined.columns.tolist()
    cols = cols[2:] + cols[:2]
    data_combined = data_combined[cols] 
    data_combined.to_csv(PROCESSED_DIR + dataset_model+".csv", index=False)
    print(data_combined)

for dataset_model in ["base", "betti","betti_der", "fl"]:
    combine_features_with_data(dataset_model)

