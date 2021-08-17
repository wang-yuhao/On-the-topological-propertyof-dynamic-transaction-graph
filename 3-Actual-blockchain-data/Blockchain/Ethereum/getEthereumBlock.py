# Read the ethereum blockchain data and extract their topological properties
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
BETTI_1_DIR = 'betti/betti_1/'

last_day = 0
step_block = 95000
start_block_number = 10427000
end_block_number   = start_block_number - step_block * 3
step_len = -1000


web3_key1 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/c338d8247a504efd85a1e8a8738bfaa7")) # yuhao2804@gmail.com 
web3_key2 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/504bf3b648ee4afa9b8ef68e82d0d5b5")) # wang.yuhao@campus.lmu.de 
web3_key3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/f9c1de855f1c43f0bcecd0fb574e8a9e")) # wangyuhaodeutsch@gmail.com 
web3_key4 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/a45e22291e964900b8caad9a15d01cad")) # yuhao199410@gmail.com
web3_key5 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/1cacc4f441a24306bb97e2b2a29f46bc")) # wyh04280030@gmail.com
web3_key6 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/329fb49ff23b433794539e4b76150529")) # wangyuh@cip.ifi.lmu.de
web3_key7 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/a8a91c9b51c249ad9356b13b742ba8e0")) # 310835246@qq.com
web3_key8 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/69742f661e374708b0bd92b8d135f724")) # 2677261148@qq.com
web3_key9 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/4bb577eede1e44efa7357325f57749c1")) # wyh940510@gmail.com
web3_key10 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/14abaf422f2649989c72f561aef24046")) # wyh04280030@qq.com

# Create a directory if it does not exist
blocks_record_path = "blocks_record.csv"

YEARS = ['2021','2020','2019','2018','2017']
for YEAR in YEARS:
    amoData_Year_dir = AMODATA_DIR + YEAR + "/"
    amoMat_Year_dir = AMOMAT_DIR + YEAR + "/"
    occMat_Year_dir = OCCMAT_DIR + YEAR + "/"
    perMat_Year_dir = PERMAT_DIR + YEAR + "/"
    betti_Year_dir = BETTI_DIR + YEAR + "/"
    betti_0_Year_dir = BETTI_0_DIR + YEAR + "/"
    betti_1_Year_dir = BETTI_1_DIR + YEAR + "/"
    
    check_dir_list = [amoData_Year_dir, amoMat_Year_dir, occMat_Year_dir, perMat_Year_dir,betti_Year_dir, betti_0_Year_dir, betti_1_Year_dir]
    for dir_name in check_dir_list:
        if not os.path.exists(dir_name):
            print("Create "+dir_name)
            os.makedirs(dir_name)


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

def save_to_file(callback_arg):
        (block_data_dict, blockNumber) = callback_arg
        tx_list = block_data_dict.values()
        #print("block_data_dict:",block_data_dict)
        print("save_to_file:",blockNumber)
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
            YEAR = getYearFromDate(eth_date)
            last_file_path = AMODATA_DIR + YEAR + "/" + str(eth_date) + ".csv"
            if os.path.exists(last_file_path):
                try:
                    eth_tx_total.to_csv(last_file_path, mode='a', header=False)
                except Exception as ex:
                    template = "An exception of type {0} occurred. Arguments: \n{1!r}"
                    message  = template.format(type(ex).__name__, ex.args)
                    print( "\n" + message)
            else:
                try:
                    eth_tx_total.to_csv(last_file_path, mode='a', header=False)
                except Exception as ex:
                    template = "An exception of type {0} occurred. Arguments: \n{1!r}"
                    message  = template.format(type(ex).__name__, ex.args)
                    print("\n" + message)
            #genMat(eth_date, eth_tx_total)
            record = "###### " + str(datetime.datetime.now()) + " last block number:"+ str(blockNumber + step_len + 1) + "\n"
            saveRecord(record)    #print(eth_tx_total)
        
        handledBlockRecord(blockNumber) 

def saveRecord(last_number):
    f = open("history.txt", "a")
    f.write(str(last_number))
    f.close()

def handleETHBlock(blockNumber):
    print("blockNumber:", blockNumber)
    if (start_block_number - blockNumber) < step_block:
        web3 = web3_key1
    elif ((start_block_number - blockNumber) >= step_block * 1) & ((start_block_number - blockNumber) < step_block * 2):
        web3 = web3_key2
    elif ((start_block_number - blockNumber) >= step_block * 2) & ((start_block_number - blockNumber) < step_block * 3):
        web3 = web3_key3
    elif ((start_block_number - blockNumber) >= step_block * 3) & ((start_block_number - blockNumber) < step_block * 4):
        web3 = web3_key4
    elif ((start_block_number - blockNumber) >= step_block * 4) & ((start_block_number - blockNumber) < step_block * 5):
        web3 = web3_key5
    elif ((start_block_number - blockNumber) >= step_block * 5) & ((start_block_number - blockNumber) < step_block * 6):
        web3 = web3_key6
    elif ((start_block_number - blockNumber) >= step_block * 6) & ((start_block_number - blockNumber) < step_block * 7):
        web3 = web3_key7
    elif ((start_block_number - blockNumber) >= step_block * 7) & ((start_block_number - blockNumber) < step_block * 8):
        web3 = web3_key8
    elif ((start_block_number - blockNumber) >= step_block * 8) & ((start_block_number - blockNumber) < step_block * 9):
        web3 = web3_key9
    else:
        web3 = web3_key10
        
    block_data_dict = {}
    while True:
        try:
            for b_number in range(blockNumber, blockNumber + step_len, -1):
                #print("b_number:",b_number)
                block = web3.eth.getBlock(b_number, True)
                block_date_unix = block.timestamp
                block_date = datetime.datetime.utcfromtimestamp(block_date_unix).strftime('%Y-%m-%d')
                txs = block.transactions

                for tx_data in iter(txs):
                    tx_hash = tx_data['hash']
                    input_addr = tx_data['from']
                    output_addr = tx_data['to']
                    tx_value = tx_data['value']
                    block_data_dict[tx_hash] = [block_date, input_addr, output_addr, tx_value]

                if (b_number % 1000) == 0:
                    print("###### Processing ETH block:",b_number, block_date)
                    
            return (block_data_dict, blockNumber)
                
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments: \n{1!r}"
            message  = template.format(type(ex).__name__, ex.args)
            print( "\n" + message)
            continue

        break
        
        
def handledBlockRecord(block_number):
    f = open(blocks_record_path, "a")
    f.write(str(datetime.datetime.now()) + "," + str(block_number) + "," + str(block_number + step_len + 1) + "\n")
    f.close()

#for start_block_number in range(start_block_number, end_block_number, step_len):
while True:
    try:

        p = Pool(10)
        for blockNumber in range(start_block_number, end_block_number, step_len):
            p.apply_async(handleETHBlock, args=(blockNumber, ), callback=save_to_file)
        p.close()
        p.join()
        print("end_block_number: ",end_block_number)

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments: \n{1!r}"
        message  = template.format(type(ex).__name__, ex.args)
        print("\n" + message)
        continue

    break