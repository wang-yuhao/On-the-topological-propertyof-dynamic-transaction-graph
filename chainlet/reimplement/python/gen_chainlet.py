# Convert input and output files to chainlet matrix, here is cluster 8.
# 1. Download input and ooutput data
# 2. Change DATA_DIR_PATH to your downloaded data path
# 3. Change the OCC_MATRIX_SIZE

import pandas as pd
import datetime
import numpy as np
import os
from os import path


def get_time(time):
    return  datetime.datetime.fromtimestamp(int(time)).strftime('%Y%m%d')

DATA_DIR_PATH = "drive/MyDrive/Colab Notebooks/Chainlet/data/"
OCC_MATRIX_SIZE = 8

processed_data_dir = DATA_DIR_PATH + "processed_data/2016/"

if os.path.exists(processed_data_dir):
        shutil.rmtree(processed_data_dir)

count_total = 0
for i in range(1, 13):
    INPUT_DATA_FILE_PATH = DATA_DIR_PATH + "raw_data/input/inputs2016_" + str(i) + ".txt"
    OUTPUT_DATA_FILE_PATH = DATA_DIR_PATH + "raw_data/output/outputs2016_" + str(i) + ".txt"
    input_pd = []
    output_pd = []
    for input_line in open(INPUT_DATA_FILE_PATH):
      #input_line = input_data.readline()
      input_line_array = input_line.split("\t")
      #print(input_line_array)
      input_line_len = int((len(input_line_array) - 2 ) / 2)
      input_line_len = input_line_len if input_line_len <= OCC_MATRIX_SIZE else OCC_MATRIX_SIZE
      input_line_time = get_time(input_line_array[0])
      input_pd.append([input_line_time, input_line_array[1], input_line_len])

    for output_line in open(OUTPUT_DATA_FILE_PATH):
      #input_line = input_data.readline()
      output_line_array = output_line.split("\t")
      #print(input_line_array)
      output_line_len = int((len(output_line_array) - 2 ) / 2)
      output_line_len = output_line_len if output_line_len <= OCC_MATRIX_SIZE else OCC_MATRIX_SIZE
      output_line_time = get_time(output_line_array[0])
      output_pd.append([output_line_time, output_line_array[1], output_line_len])

    #output_data = pd.read_csv(OUTPUT_DATA_FILE_PATH)
    input_df = pd.DataFrame(input_pd, columns=["time", "hash_index","input_occ"])
    output_df = pd.DataFrame(output_pd, columns=["time", "hash_index","output_occ"])
    merge_df = pd.merge(input_df, output_df, how='inner', on=['time', 'hash_index'])
    print(input_df)
    print(output_df)
    print(merge_df)
    matrix_name = merge_df["time"].drop_duplicates().reset_index(drop=True)


    for n in range(matrix_name.size):
      matrix_day = merge_df.loc[merge_df.loc[:,"time"]==matrix_name[n], :]
      matrix_day_short = matrix_day.loc[:,"input_occ":"output_occ"]
      matrix_day_short["character"]  = 0
      matrix_day_short = matrix_day_short.groupby(['input_occ', 'output_occ'], as_index=False)['character'].count()

      #matrix_day_short['character'] = matrix_day_short['character'].replace('',0)
      print("matrix_day_short: ",matrix_day_short)

      matrix_init = np.zeros((OCC_MATRIX_SIZE,OCC_MATRIX_SIZE), dtype=np.int)
      for i in range(1,9):
        for j in range(1,9):
          if matrix_day_short.loc[(matrix_day_short['input_occ']==i) & (matrix_day_short['output_occ']==j), 'character'].values.size > 0:
            matrix_init[i-1][j-1] = int(matrix_day_short.loc[(matrix_day_short['input_occ']==i) & (matrix_day_short['output_occ']==j), 'character'].values)
      print("matrix_init: ",matrix_init)

      if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
      matrix_file_name = processed_data_dir + matrix_name[n] + ".csv"
      if path.exists(matrix_file_name):
        print("*******************"*3)
        print("{} already exists!".format(matrix_file_name))
        print("matrix_init: ", matrix_init)
        matrix_read = pd.read_csv(matrix_file_name,delimiter=',', header=None).to_numpy()
        print("matrix_read: ", matrix_read)
        matrix_sum = np.add(matrix_init, matrix_read)
        np.savetxt(matrix_file_name, matrix_sum, fmt='%i', delimiter=",")
        print("matrix_sum has been saved: ", matrix_sum)
      else:
        np.savetxt(matrix_file_name, matrix_init, fmt='%i', delimiter=",")
        #print("matrix_init has been saved: ", matrix_init)

      count_total = count_total + matrix_init.sum()
    print("count_total: ",count_total)

print("count_total final: ",count_total)