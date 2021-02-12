def MatrixJustification():
    data_dir = args[0]
    coins = ["Bitcoin", "Litecoin", "Namecoin"]
    for coin in coins:
        totalTx = 0;
        matrix  = np.zeros((100,100))
        iMax = 0
        oMax = 0
        for year in range(2009, 2018):
            for day in range(366):
                file_name = data_dir + coin + "/createddata/daily/" + year + "_" + day + ".txt"
                br = open(file_name, "r")
                
                for line in br.readline():
                    totalTx = totalTx + 1
                    split = line.split("\t")
                    split_input = int(split[2])
                    split_output = int(split[3])
                    if (split_input > iMax):
                        iMax = split_input
                    if(split_output > oMax):
                        oMax = split_output
                        
                    if(split_input > len(matrix)):
                        split_input = len(matrix)
                    if(split_output > len(matrix)):
                        split_output = len(matrix)
                        
                    matrix[split_input - 1][split_output - 1] = matrix[split_input - 1][split_output - 1]  + 1
                
        for i in range(2, 50):
            subTotal = 0
            for j in range(0, i):
                for k in range(0, i):
                    subTotal = subTotal + matrix[j][k]
        
        print(coin + " : " + totalTx)
        
        for i in range(0, 21):
            for j in range(0, 21):
                print(100 * matrix[i][j] / totalTx + "\t")
                
                