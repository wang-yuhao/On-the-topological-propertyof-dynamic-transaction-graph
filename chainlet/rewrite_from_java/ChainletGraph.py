class ChainletGraph:
    def __init__(self):
        self.timePeriodMax = 366
    
    def getmatrix(self, coin):
        data_dir = "H:/data/" + coin + "/createddata/daily/"
        chDir = "H:/data/" + coin + "/createddata/Price.ChainletGraph/"
        
        if(os.path.isdir(chDir)):
            shutil.rmtree(chDir)

        
        try: 
            os.makedirs(chDir)
        except OSError:
            print ("Creation of the directory %s failed" % chDir)
        else:
            print ("Successfully created the directory %s" % chDir)
            
        max_count = 0
        
        for year in range(2010, 2018):
            for timePeriod in range(1, self.timePeriodMax):
                dim = 20
                grMat = {}
                file_name = data_dir + str(year) + "_" + str(timePeriod) + ".txt"
                inBr = open(file_name, "r")
                for line in inBr.readline():
                    arr = lne.split("\t")
                    icount = int(arr[2])
                    ocount = int(arr[3])
                    if (icount > dim):
                        icount = dim
                    if (ocount > dim):
                        ocount = dim
                    chId = "X" + str(icount) + ":" + str(ocount)
                    if(chId not in grMat):
                        grMat[chId] = {}
                    
                    amount = int(arr[4])
                    if(amount > max):
                        max = amount
                        print("new max: ", max)
                    
                    chAmounts = grMat.get(chId)
                    if(amount not in chAmouts):
                        chAmounts[amount] = 0
                    chAmounts[amount] = chAmounts.get(amount) + 1
                    
                inBr.close()
                
                writeMatrix(year, timePeriod, chDir, grMat)
                
                
    def writeMatrix(self, year, timePeriod, chDir, grMat):
        wr = open(str(chDir) + str(year) + str(x) + str(timePeriod) + ".csv", "w")
        bf = ""
        for i in range(1,21):
            for o in range(1,21):
                chId = "x" + str(i) + ":" + str(o)
                if(chId in grMat):
                    for l in grMat.get(chId).keys():
                        bf= bf + l + "\t" + grMat.get(chId).get(l) + "\t"
                else: 
                    bf = bf + "0\t0"
                bf = bf + "\r\n"
                
        wr.write(bt)
                

chainlet = ChainletGraph()
coins = ["Namecoin"]
for coin in coins:
    chainlet.getmatrix(coin)