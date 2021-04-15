
"""
# Read year_day_hour.txt into occM and amoM
icount: input count
ocount: output count
# occM:
update: occM[icount-1][ocount - 1] = occM[icount-1][ocount - 1] + 1
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]] (20, 20)

amoM:
update: amoM[icount-1][ocount - 1] = amoM[icount-1][ocount - 1] + int(arr[4])
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]] (20, 20)
"""

import os
import shutil
import numpy as np

def getmatrix(coin):
    timePeriodMax = 366
    data_dir = "H:/data/" + coin + "/createddata/hourly/"
    if(os.path.isdir(data_dir)):
        shutil.rmtree(data_dir)

        
    try: 
        os.makedirs(data_dir)
    except OSError:
        print ("Creation of the directory %s failed" % data_dir)
    else:
        print ("Successfully created the directory %s" % data_dir)
    
    occDir = "H:/data/" + coin + "/createddata/hourlyOccMatrices/"
    amoDir  = "H:/data/" + coin + "/createddata/hourlyAmoMatrices/"
    
    checkTxTimePeriod(timePeriodMax)
    wrinfo = open(data_dir+"info.txt", "a")
    
    for year in range(2009, 2018):
        for day in range(1, timePeriodMax+1):
            for hour in range(0,24):
                dim = 20
                #occM = [[0 for i in range (dim)] for j in range(dim)]
                occM = np.zeros((dim,dim))
                amoM = np.zeros((dim,dim))
                print("occM: ",occM, occM.shape)
                
                fileName = data_dir + str(year) +"_" + str(day) + "_" + str(hour) + ".txt"
                inBr = open(fileName, "w+")
                
                i = 0
                transition = 0
                merge = 0
                split = 0
                for line in inBr.readline():
                    arr = line.split("\t")
                    icount = int(arr[2])
                    ocount = int(arr[3])
                    if(icount == ocount):
                        transition = transition + 1
                    elif(icount  > ocount):
                        merge = merge + 1
                    else:
                        split = split + 1
                    
                    if (icount > dim):
                        icount = dim
                    if (ocount > dim):
                        ocount = dim
                    
                    occM[icount-1][ocount - 1] = occM[icount-1][ocount - 1] + 1
                    amoM[icount-1][ocount - 1] = amoM[icount-1][ocount - 1] + int(arr[4])

                inBr.close()
                
                total = merge + split + transition
                
                wrinfo.write(str(year) + "\t" + str(day) + "\t" + str(total) + "\t" + str(merge / total) + "\t" + str(split / total) + "\t" + (transition / total) + "\r\n")
                
                if (total > 0):
                    writeMatrix(occM, occDir, "occ" + str(year) + "_" + str(day) + "_" + str(hour))
                    writeMatrix(amoM, amoDir, "amo" + str(year) + "_" + str(day) + "_" + str(hour)
                
def writeMatrix(matrix, data_dir, file_name):
    np.savetxt(data_dir+file_name+".csv", matrix, delimiter=",")
                                
def checkTxTimePeriod(timePeriodMax):
    if((timePeriodMax != 52) & (timePeriodMax != 366)):
        print("time period is unknown. Should be day or week.")
    return

coins = ["Bitcoin", "Litecoin", "Namecoin"]
for coin in coins:
    getmatrix(coin)