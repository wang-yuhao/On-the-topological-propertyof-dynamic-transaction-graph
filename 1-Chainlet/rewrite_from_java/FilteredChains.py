import os
import numpy as np
import math
class FilteredChains:
    def __init__(self):
        self.timePeriodMax = 366
        coins = ["Bitcoin", "Litecoin", "Namecoin"]
        for coin in coins:
            print(coin + "processing....")
            getmatrix(coin)
        
    def getmatrix(self, coin):
        data_dir = "H:/data/" + coin + "/createddata/daily/"
        filter_dir = "H:/data/" + coin + "/createddata/filteredDailyOccMatrices/"
        
        os.mkdirs(filter_dir)
        
        for year in range(2009, 2018):
            for timePeriod in range(self.timePeriodMax):
                fileName = data_dir + str(year) + "_" + str(timePeriod) + ".txt"
                inBr = open(fileName, "r")
                content = []
                for line in inBr.readline():
                    content.append(line)
                
                inBr.close()
                
                for filterIndex in range(0, 100, 10):
                    dim = 20
                    has = False
                    threshold = filterIndex
                    occM = np.zeros((dim,dim))
                    
                for l in content: 
                    arr = l.split("\t")
                    icount = int(arr[2])
                    ocount = int(arr[3])
                    amount = float(arr[4] / math.pow(10,8))
                    if amount > threshold:
                        if(icount > dim):
                            icount = dim
                        if(ocount > dim):
                            ocount = dim
                        occM[icount - 1][ocount - 1] = occM[icount - 1][ocount - 1] + 1
                        has = True
                    
                if(has):
                    np.savetxt(filter_dir+"occ" + str(year) + "_" + str(timePeriod) + "_" + str(filterIndex) +  ".csv", occM, delimiter=",")
                    
fChain = FilteredChains()