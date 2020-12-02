'''
In order to generate Betti_0 and betti_1 of 2017 dailyAmountMatrices, change the format of all matrices according the format of the http://people.maths.ox.ac.uk/nanda/perseus/
    format: 
        3: the ambient dimension, i.e., the number of coordinates per vertex.
        1 0.01 100: the radius scaling factor k=1, the step size s=0.01, the number of steps N=100
        1.2 3.4 -0.9 0.5: the vertex (1.2, 3.4, -0.9) with associated radius r = 0.5
        2.0 -6.6 4.1 0.3: the vertex (2.0, -6.6, 4.1) with associated radius r = 0.3
        and so on! 
    example:
        (http://people.maths.ox.ac.uk/nanda/source/distmat.txt)

then use the following command to convert matrix:
    (path to perseus executable) (complex type) (input filename) (output file string)
    
command example:
    ./perseus distmat ../data/dailyAmoMatrices/amo2017001.csv ../data/random_regression

'''



import pandas as pd
import os, shutil
import numpy as np

YEAR = 2017
input_data_path = "C:/Users/wang.yuhao/Documents/ChainNet/data/original_data/dailyAmoMatrices/"
output_data_path = "C:/Users/wang.yuhao/Documents/ChainNet/data/processed_data/dailyVrAmoMatrices/"

def clean_folder(folder_name):
    for filename in os.listdir(folder_name):
        file_path = os.path.join(folder_name, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print("Remove {} successful.".format(file_path) )
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def read_csv(file_name, day):
    names=[]
    for i in range(20):
        names.append(str(i))
    data = pd.read_csv(file_name, header=None, names=names)
    data = data/(10^8) + 1
    data = data.apply(np.log)
    row_count = pd.DataFrame({"0": [data.shape[0]]})
    param = pd.DataFrame({"0": [0], "1": [1], "2": [101], "3": [2]})
    header =  row_count.append(param, ignore_index=True)
    data = header.append(data, ignore_index=True)
    data.to_csv(output_data_path +  "vrAmo" + str(YEAR) + '{:03}'.format(day) + ".csv", sep=" ", index=False, header=False)
    if(day % 10 == 0):
        print("The data conversion on the {}th day was successful.".format(day) )


clean_folder(output_data_path)   
for day in range(1, 366):
    file_name = input_data_path + "amo" + str(YEAR) + '{:03}'.format(day) + ".csv"
    read_csv(file_name, day)