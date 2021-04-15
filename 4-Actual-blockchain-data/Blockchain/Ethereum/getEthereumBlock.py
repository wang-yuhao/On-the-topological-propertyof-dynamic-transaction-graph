# Read the bitcoin blockchain data and extract their topological properties
# Modify on 12.04.2021

from web3 import Web3 
from collections import defaultdict
from multiprocessing import Pool, Queue
import multiprocessing
import datetime
import math
import time
import pandas as pd
import requests
import numpy as np
import json
import os
import torch

AMODATA_DIR = 'amoData/'
AMOMAT_DIR = 'amoMat/'
OCCMAT_DIR = 'occMat/'
PERMAT_DIR = 'perMat/'
BETTI_DIR = 'betti/'
BETTI_0_DIR = 'betti/betti_0/'
BETTI_1_DIR = 'betti_1/'

last_day = 0
start_block_number = 12160000
end_block_number = 4200000
step_len = -20000

web3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/c338d8247a504efd85a1e8a8738bfaa7"))

def getBetweenDay(begin_date, end_date):
    date_list = []
    date_arr = []
    date_unix_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    print(type(begin_date))
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

def new_file(dir):
    list = os.listdir(dir)
    list.sort(key=lambda fn:os.path.getmtime(dir+fn))
    filetime = datetime.datetime.fromtimestamp(os.path.getmtime(dir+list[-1]))
    filepath = os.path.join(dir, list[-1])
    print("latest file is: ", list[-1])
    print("time: ", filetime .strftime("%Y-%m-%d %H:%M:%S"))
    return filepath

# Read block
def read_block(block):
    tx_btc_total = []
    print("block:", block['hash'])
    # Get every transaction data
    tx_api = 'https://blockchain.info/rawblock/'+block['hash']
    tx_data = requests.get(tx_api)
    None_count = 0
    for tx in tx_data.json()['tx']:
        # Extract its input size and output size
        # chain_data.append([tx['vin_sz'], tx['vout_sz']])
        vin = tx['vin_sz']
        vout = tx['vout_sz']
        if vin > 20:
            vin = 20
        if vout > 20:
            vout = 20
        IOName = f'{vin:02}' + f'{vout:02}'
        tx_value = 0
        for value in tx['inputs']:
            if ('prev_out' in value) & (value['prev_out'] is not None):
                #print("value:", value)
                tx_value = tx_value + value['prev_out']['value']
            else:
                None_count = None_count + 1
        tx_btc_total.append([IOName, tx_value])
    tx_btc_total = pd.DataFrame(tx_btc_total)
    #print("None_count: ", None_count)
    return tx_btc_total
    # print(tx_btc_total)
    
# create a IO-Name list 
def create_IONameList():
    IONameList = []
    for i in range(1, 21):
        for j in range(1, 21):
            IOName = f'{i:02}' + f'{j:02}'
            IONameList.append(IOName)
    return IONameList

# Merge two dictionaries
def mergeDict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
               dict3[key] = [value , dict1[key]]
    return dict3

# with 10-quantiles
# calculate the quantile of two nodes
def calculate_quantile(amount_1, amount_2):
    # for quantile_ in range(quantile_value,1,quantile_value)
    # print("quantile;",amount_1.quantile([0.25, 0.5, 0.75])[0][0.25])
    quantile_percentage = [ i/500 for i in range(1, 500, 1)]
    quantile_squar = (amount_1.quantile(quantile_percentage)[0]-amount_2.quantile(quantile_percentage)[0])**2   
    quantile_sum = quantile_squar.sum()
    # print(quantile_1, quantile_2, quantile_3)
    return (quantile_sum)**0.5

def getYearFromDate(date):
    Year = date.split("-")[0]
    return Year

def getYearsFromDate(begin_date, end_date):
    begin_Year = begin_date.split("-")[0]
    end_Year = end_date.split("-")[0]
    YEARS = [str(i) for i in range(int(begin_Year), int(end_Year) + 1)]
    return YEARS 

def handelETHBlock(blockNumber,return_dict):
    block = web3.eth.getBlock(blockNumber, True)
    block_date_unix = block.timestamp
    block_date = datetime.datetime.utcfromtimestamp(block_date_unix).strftime('%Y-%m-%d')
    txs = block.transactions

    for tx_data in iter(txs):
        tx_hash = tx_data['hash']
        input_addr = tx_data['from']
        output_addr = tx_data['to']
        tx_value = tx_data['value']
        return_dict[tx_hash] = [block_date, input_addr, output_addr, tx_value]
    
    print("###### Processing ETH block:",blockNumber, block_date)

def saveRecord(last_number):
    f = open("history.txt", "w")
    f.write(str(last_number))
    f.close()

# If any errors are encountered, it will automatically restart.
def genMat(date_unix, amo_data_total):
    time_now = time.time()
    Year = getYearFromDate(date_unix[0])

    while True:
        try:
                print("check file {}...".format(date_unix[0]+".json"))
                if 'occ' + date_unix[0] + '.csv' in os.listdir(OCCMAT_DIR + YEAR + "/"):
                    print(date_unix[0]+".csv already exists ...")
                    # continue
                else:
                    # Get the daily block
                    # amo_data_total = pd.read_csv(AMODATA_DIR + YEAR + "/" + date_unix[0] + ".csv", index_col=0, converters={"0":str})
                    amo_data_total.to_csv(AMODATA_DIR + YEAR + "/" + date_unix[0] + ".csv")
                    amo_data_total.columns = ['IOSize', 'tx_value']
                    amo_data_total["tx_value_log"] = amo_data_total["tx_value"].map(lambda x: round(math.log(1 + x/(10**8)),5))
                    amo_data_total.reset_index(drop=True)
                    amo_data_total_dict = amo_data_total.groupby('IOSize').tx_value_log.apply(list).to_dict()
                    IONameList = create_IONameList()
                    print("amo_data_total:", amo_data_total)
                    MATRIX_SIZE = len(IONameList)
                    amoMat = [[0] * MATRIX_SIZE] * MATRIX_SIZE
                    amoMat_df = pd.DataFrame(amoMat, columns = IONameList, index = IONameList)


                    for IO_1 in IONameList:
                        if IO_1 in amo_data_total_dict:
                            amount_1 = pd.DataFrame(amo_data_total_dict[IO_1])
                        else:
                            amount_1 = pd.DataFrame([0])
                        for IO_2 in IONameList:
                            if IO_2 in amo_data_total_dict:
                                amount_2 = pd.DataFrame(amo_data_total_dict[IO_2])
                            else:
                                amount_2 =  pd.DataFrame([0])
                            amoMat_df.loc[IO_1, IO_2] = calculate_quantile(amount_1, amount_2)
                        #print("amoMat_df:", amoMat_df)
                    print("amoMat_df:", amoMat_df)

                    # Calculate betti nummber
                    # add parameter for perseus computing 
                    amoMat_df.apply(str)
                    param_1 = pd.DataFrame([["400"]], columns=["0101"])
                    param_2 = pd.DataFrame([["1","1","101","1"]], columns=["0101", "0102", "0103", "0104"])
                    param_amoMat_df = pd.concat([param_1,param_2, amoMat_df], axis=0, sort=False)
                    perMat_path = PERMAT_DIR + YEAR + "/" + date_unix[0] + ".csv"
                    param_amoMat_df.to_csv(perMat_path, index=False, sep='\t', header=False)

                    # use perseus to compute betti number
                    betti_path = "betti/" + YEAR + "/" + date_unix[0]
                    betti_0_path = "betti/betti_0/" + YEAR + "/" + date_unix[0] + "_betti_0.csv"
                    betti_1_path = "betti/betti_1/" + YEAR + "/" + date_unix[0] + "_betti_1.csv"
                    perseus_command = "perseus/perseus distmat " + perMat_path + " " + betti_path
                    if(os.system(perseus_command) == 0):
                        betti_number = pd.read_csv(betti_path +"_betti.txt", sep='\s+', index_col=0, names=["betti_0", "betti_1"])
                        init_betti_0 = pd.DataFrame([[0]]*101, columns=["betti_0"])
                        init_betti_1 = pd.DataFrame([[0]]*101, columns=["betti_1"])
                        betti_0 = (betti_number["betti_0"] + init_betti_0["betti_0"]).fillna(axis=0, method='ffill').fillna(0).astype(int)
                        betti_1 = (betti_number["betti_1"] + init_betti_1["betti_1"]).fillna(axis=0, method='ffill').fillna(0).astype(int)
                        betti_0.to_csv(betti_0_path)
                        betti_1.to_csv(betti_1_path)
                        print("Successfully calculated Betti number!")
                    else:
                        print("Failed to calculate Betti number!")

                    # Calculate OccMat and AmoMat
                    io_data_amo = amo_data_total['tx_value'].groupby(amo_data_total['IOSize']).sum()
                    io_data_occ = amo_data_total.groupby(amo_data_total['IOSize']).count()
                    io_data_occ = io_data_occ.iloc[:,1]
                    occMat = torch.zeros(20,20)
                    amoMat = torch.zeros(20,20)
                    for i in range(1,21):
                        for j in range(1,21):
                            io_name = str(i).zfill(2) + str(j).zfill(2)
                            if(io_name in io_data_amo.index):
                                amoMat[i-1][j-1] = io_data_amo[io_name]
                            if(io_name in io_data_occ.index):
                                occMat[i-1][j-1] = io_data_occ[io_name]

                    amoMat_np = amoMat.numpy()
                    amoMat_df = pd.DataFrame(amoMat_np)
                    #amoMat_df.to_csv(AMOMAT_DIR + 'amo2020' + str(day).zfill(3) + '.csv', float_format='%.0f', header=False, index=False)
                    amoMat_df.to_csv(AMOMAT_DIR + YEAR + "/" + 'amo' + date_unix[0] + '.csv', float_format='%.0f', header=False, index=False)
                    occMat_np = occMat.numpy()
                    occMat_df = pd.DataFrame(occMat_np)
                    #occMat_df.to_csv(OCCMAT_DIR + 'occ2020' + str(day).zfill(3) + '.csv', float_format='%.0f', header=False, index=False)
                    occMat_df.to_csv(OCCMAT_DIR + YEAR + "/" + 'occ' + date_unix[0] + '.csv', float_format='%.0f', header=False, index=False)
                
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments: \n{1!r}"
            message  = template.format(type(ex).__name__, ex.args)
            print(date_unix[0] + "\n" + message)
            continue
            
        break
        
    # count the time
    total_days = len(getBetweenDay(begin_date, end_date))
    finished_days = len(os.listdir(BETTI_0_DIR + YEAR + "/")) - 1 
    left_days = total_days - finished_days
    finished_percentage = math.floor(finished_days / total_days * 100)
    single_file_time = time.time()-time_now
    left_time = single_file_time * left_days
    print('\tcost: {:.4f}s/file; left time: {:.4f}s; {} {}%'.format(single_file_time, left_time,  "#"*finished_percentage+"."*(100-finished_percentage), finished_percentage))                


for start_block_number in range(start_block_number, end_block_number, step_len):
    if(os.system("rm -rf ./amoData/.ipynb_checkpoints") == 0):
        last_day = min(os.listdir("./amoData/")).split(".")[0]

    print("start_block_number:",start_block_number)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()     
    p = Pool(4)
    for blockNumber in range(start_block_number, start_block_number + step_len, -1):
        p.apply_async(handelETHBlock, args=(blockNumber, return_dict))
    p.close()
    p.join()

    tx_list = return_dict.values()
    #print(tx_list)
    tx_df = pd.DataFrame(tx_list, columns=["block_date","input_addr", "output_addr", "value"])

    for eth_date in tx_df["block_date"].groupby(tx_df["block_date"]).count().index.values:
        daily_input_df = tx_df.loc[tx_df["block_date"]==eth_date].loc[:,["input_addr","value"]]
        daily_input_df["in_sz"] = 1

        daily_output_df = tx_df.loc[tx_df["block_date"]==eth_date].loc[:,["output_addr","value"]]
        daily_output_df["out_sz"] = 1

        daily_in_data = pd.concat([daily_input_df["in_sz"].groupby(daily_input_df["input_addr"]).sum(), daily_input_df["value"].groupby(daily_input_df["input_addr"]).sum()], axis=1)
        daily_out_data = pd.concat([daily_output_df["out_sz"].groupby(daily_output_df["output_addr"]).sum(), daily_output_df["value"].groupby(daily_output_df["output_addr"]).sum()], axis=1)

        daily_in_data.index.name="addr"
        daily_out_data.index.name="addr"
        daily_in_data["out_sz"] = 1
        daily_in_data["in_sz"] = daily_in_data["in_sz"] + 1

        daily_out_data["in_sz"] = 1
        daily_out_data["out_sz"] = daily_out_data["out_sz"] + 1

        IO_SZ = daily_out_data.add(daily_in_data, fill_value=0)

        eth_tx_total = []
        for tx in IO_SZ.values:
            # Extract its input size and output size
            vin = int(tx[0])
            vout = int(tx[1])
            if vin > 20:
                vin = 20
            if vout > 20:
                vout = 20
            IOName = f'{vin:02}' + f'{vout:02}'
            tx_value = tx[2]
            eth_tx_total.append([IOName, tx_value])
        eth_tx_total = pd.DataFrame(eth_tx_total, columns=["IO_SZ", "value"])
        if (last_day != 0) & (last_day == eth_date):
            last_file_path = "./amoData/" + last_day + ".csv"
            last_day_blocks = pd.read_csv(last_file_path, index_col = 0, converters={"0":str})
            last_day_blocks.columns = ["IO_SZ", "value"]
            eth_tx_total = pd.concat([last_day_blocks, eth_tx_total], axis=0).reset_index(drop=True)
            
	genMat(eth_date, eth_tx_total)

    saveRecord(start_block_number-20000)    #print(eth_tx_total)

