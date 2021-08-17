## How to obtain the topological properties of the Bitcoin blockchain?

1. First download this repository, and then run the following command under this folder:

```
Unzip perseus.zip
python3.7 getBitcoinBlock.py
```

Next, you can change the following dates in `getBitcoinBlock.py` to generate data for the required period.

```
begin_date ='2020-01-01'
end_date ='2020-12-31'
```

This command will build the following folder structure:

```
.
|--amoData	# The "input", "output" and "value" data of blocks
|--amoMat	# Amount Matrix
   |--2020
      |--amo2020001.csv
      |--...
      |--amo2020366.csv
|--occMat	# Occurence Matrix
   |--2020
      |--occ2020001.csv
      |--...
      |--amo2020366.csv
|--betti	# Betti number, including betti_0 and betti_1
   |--2020
      |--...
   |--betti_0
      |--...
   |--betti_1
      |--...
|--perMat	# Matrix for perseus process
   |--2020
      |--...
```


2. Or you can directly download the bitcoin_blockchain_2020.zip, it including:

```
bitcoin_blockchain_2020
|
|--amoMat	# Amount Matrix
   |--2020
      |--amo2020001.csv
      |--...
      |--amo2020366.csv
|--occMat	# Occurence Matrix
   |--2020
      |--occ2020001.csv
      |--...
      |--amo2020366.csv
|--2020_betti_0.csv	# 2020 betti number: betti_0
|--2020_betti_1.csv	# 2020 betti number: betti_1
```

Bitcoin Blockchain source: [BLOCKCHAIN](https://www.blockchain.com/api/blockchain_api)
Bitcoin price sources: [BLOCKCHAIN](https://www.blockchain.com/charts/market-price)
Bitcoin transaction: Obtained by merging amoData. 
Bitcoin transaction(secondary source): https://www.blockchain.com/charts/n-transactions.
 

