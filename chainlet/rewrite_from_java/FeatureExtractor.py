class FeatureExtractor:
    def __init__(self):
        preprocess = False
        data_dir = "H:/data/createddata/feature/"
        os.mkdirs(data_dir)
        years = {2016, 2017}
        
        for year in years:
            splitFiles(year)
        print("year\tday\tmeanValue\tmedianValue\thoMedian\tmeanDegree\tmedianDegree\taddCount\ttxCount")
        
        for year in years:
            for day in range(366):
                br = open(data_dir + year + "_" + day + ".txt", "r")
                # DescriptiveStatistics amounts = new DescriptiveStatistics();
                amounts = []
                hourlyTx = {}
                inDegrees = {}
                outDegrees = {}
                addresses = []
                for line in br.readline():
                    arr = line.split("\t")
                    prefix = arr[0]
                    tx = arr[2]
                    time = int(arr[1])
                    blockDate = datetime.datetime.fromtimestamp(time)
                    thishour = blockDate.hour
                    if(thishour not in hourlyTx):
                        hourlyTx[thishour] = 0
                    hourlyTx[thishour] = hourlyTx[thishour] + 1
                    
                    if(prefix.lower() == "i"):
                        inDegrees[tx] = (len(arr) - 3) / 2
                    elif(prefix.lower() == "o"):
                        outDegrees[tx] = (len(arr) - 3) / 2
                        amount = 0
                        for i in range(3, len(arr) - 1, 2):
                            amount = amount + int(arr[i+1])
                            # hashset?
                            addresses.append(arr[i])
                        amounts.append(amount)
                statistics_amount = np.array(amounts)        
                meanValue = np.mean(statistics_amount)
                medianValue = np.percentile(statistics_amount, 50)
                hotx = []
                for v in hourlyTx.values():
                    hoTx.append(v)
                    
                statistics_hoTx = np.array(hoTx)
                hoMedian = np.percentile(statistics_hoTx, 50)
                
                degrees = []
                for tx in inDegrees.keys():
                    if (tx in outDegrees):
                        degree = inDegrees[tx]
                        for f in range(1,outDegrees[tx]):
                            degrees.append(degree)
                            
                meanDegree = np.mean(degrees)
                medianDegree = np.percentile(degrees, 50)
                addCount = len(addresses)
                txCount = len(inDegrees)
                
                print(str(year) + "\t" + str(day) + "\t" + str(meanValue) + "\t" + str(medianValue) + "\t" + str(hoMedian) + "\t" + str(meanDegree) + "\t" + str(medianDegree) + "\t" + str(addCount) + "\t" + str(txCount))
                            
                
        
    def splitFiles(refYear):
        content = {}
        
        # read input and output data from these files
        f = ["H:/data/createddata/txInputs.txt", "H:/data/createddata/txOutputs.txt"]
        for fileName in f:
            substring = fileName.substring(26,27)
            inBr = open(fileName, "r")
            txIds = {}
            line = ""
            l = 0
            for line in inBr.readline():
                l = l + 1
                if(l % 100000 == 0):
                    print("l: ", l)

                
                if(len(line) < 10):
                    continue
                
                arr = line.split("\t")
                time = int(arr[0])
                blockDate = datetime.datetime.fromtimestamp(time)
                
                year = blockDate.year
                if(year == refYear):
                    tx = arr[1]
                    day = blockDate.getDayOfYear()
                    if(day not in content):
                        content[day] = ""
                    content[day] = content[day] + substring + "\t" + line + "\r\n"
                    if(len(content[day]) > 100000):
                        write(year, day, content[day])
                        content.pop(day, None)
                        
        for c in content.keys():
            write(refYear, c, content[c])
        
    def write(year, day, stringBuffer):
        wr = open("H:/data/createddata/feature/" + year + "_" + day + ".txt", "a")
        wr.write(stringBuffer)
        
            