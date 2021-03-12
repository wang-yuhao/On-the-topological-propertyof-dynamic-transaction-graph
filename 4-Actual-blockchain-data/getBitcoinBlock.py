import datetime
import math
import time
import pandas
import requests
import numpy as np
import json
import os

# set the data output directory
data_dir = 'drive/MyDrive/Colab Notebooks/bitcoin/'
# set the data start time
begin_datum = '2020-02-12'

def getBetweenDay(begin_date):
    date_list = []
    date_arr = []
    date_unix_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    print(type(begin_date))
    print("begin_date:",begin_date)
    end_date = datetime.datetime.strptime(time.strftime('%Y-%m-%d', time.localtime(time.time())), "%Y-%m-%d")
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

# If any errors are encountered, it will automatically restart.
for i in range(0,1000):
    while True:
        try:  
            for datum_unix in getBetweenDay(begin_datum):
                print("check file {}...".format(datum_unix[0]+".json"))
                if datum_unix[0]+".json" in os.listdir(data_dir):
                    print(datum_unix[0]+".json already exists ...")
                    continue
                else:
                    begin_datum = datum_unix[0]
                    print("begin_datum:", begin_datum)

                    # Get the daily block
                    datum = "https://blockchain.info/blocks/"+datum_unix[1]+"?format=json"
                    res = requests.get(datum)
                    chain_data = []
                    for block in res.json()["blocks"]:
                        print("block:", block['hash'])

                        # Get every transaction data
                        tx_api = 'https://blockchain.info/rawblock/'+block['hash']
                        tx_data = requests.get(tx_api)
                        for tx in tx_data.json()['tx']:

                            # Extract its input size and output size
                            chain_data.append([tx['vin_sz'], tx['vout_sz']])

                    # If there are more transactions coming in this daily block.
                    next_block_unix = str(math.trunc(float(datum_unix[1]) + 86400000))
                    print("next_block_unix:",next_block_unix)
                    next_block = "https://blockchain.info/blocks/"+next_block_unix+"?format=json"
                    while not (requests.get(next_block)):
                        # Wait 10 minutes 
                        time.sleep(600)
                        begin_datum = datum_unix[0]
                        print("begin_datum:", begin_datum)
                        datum = "https://blockchain.info/blocks/"+datum_unix[1]+"?format=json"
                        res = requests.get(datum)
                        chain_data = []
                        for block in res.json()["blocks"]:
                            print("block:", block['hash'])
                            tx_api = 'https://blockchain.info/rawblock/'+block['hash']
                            tx_data = requests.get(tx_api)
                            for tx in tx_data.json()['tx']:
                                chain_data.append([tx['vin_sz'], tx['vout_sz'], ])

                    # save this daily block data into a %Y-%m-%d.json file 
                    with open(data_dir + datum_unix[0] + '.json', 'w') as outfile:
                            json.dump(chain_data, outfile)
                            print("check file {}...".format(datum_unix[0]+".json"))

        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments: \n{1!r}"
            message  = template.format(type(ex).__name__, ex.args)
            print("\n"+message)
            continue
        break