from collections import defaultdict

def JungMiner():
    graph = defaultdict(list)
    
    # change the data_dir and crdir
    data_dir = "C:\\Projects\\Coin\\bitcoin_dataset\\"
    crdir = data_dir+"createddata\\"
    for year in range(2011, 2015):
        for week in range(1, 53):
            dim  = 20
            occM = np.zeros((dim,dim))
            sumM = np.zeros((dim, dim))
            ig = defaultdict(list)
            inBr = open(data_dir + "txInAll.txt")
            outBr = open(data_dir + "txOutAll.txt")
            
            inTranx = defaultdict(list)
            outTranx = defaultdict(list)
            
            sums = {}
            
            loadTranx(inBr, year, week, inTranx, sums)
            loadTranx(outBr, year, week, outTranx, sums)
            
            sIn = len(inTranx)
            sOut = len(outTranx)
            
            res = [key for key in outTranx.keys() if key in inTranx.keys]
            sSec = len(outTranx)
            
            inBr.close()
            outBr.close()
            
            transition = 0
            merge = 0
            split = 0
            for txId in inTranx.keys():
                icount = len(inTranx[txId])
                if(txid in outTranx):
                    ocount = len(outTranx[txId])
                    if(icount == ocount):
                        transition = transition + 1
                    elif(icount > ocount):
                        merge = merge + 1
                    else:
                        split = split + 1
                    if(icount> dim):
                        icount= dim
                    if(ocount> dim):
                        ocount= dim
                    occM[icount-1][ocount-1] = occM[icount-1][ocount-1] + 1
                    sumM[icount-1][ocount-1] = sumM[icount-1][ocount-1] + sums.get(txId)/Math.pow(10,5)
                    
            print(year+" "+week+" "+sIn + " " + sOut+" "+sSec+" Merge:"+merge+ " Split:"+split+" Transition:"+transition)
            if(merge+split+transition>0){
                occM = np.array(occM)
                sumM = np.array(sumM)
                np.savetxt(filter_dir+"occ" + str(year) + "week"+week+".csv", occM, delimiter=",")
                np.savetxt(filter_dir+"sum" + str(year) + "week"+week+".csv", sumM, delimiter=",")
                #writeMatrix(year,week,occM,crdir,"occ");
                #writeMatrix(year,week,sumM,crdir,"sum");
            }
            
            

def loadTranx(inBr, year, week, inTranx, sums):
    for line in inBr.readline():
        arr = line.split("\t")
        txId = int(arr[0])
        time = int(arr[1])
        time = datetime.datetime.fromtimestamp(time)
        txYear = time.year
        
        txWeek = time.week
        
        if((year == txYear) & (week == txWeek)):
            adds = arr[3].split(",")
            inTranx[txId] = []
            for a in adds:
                inTranx[txId].append(a)
            
            sums[txId] = arr[2]
        
            