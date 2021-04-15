## How to obtain the topological properties of the Ethereum blockchain?

1. First download this repository, and then run the following command under this folder:

```
Unzip perseus.zip
python3.7 getEthereumBlock.py
```

Next, you can change the following block numbers in `getEthereumBlock.py` to generate data for the required period.

```
start_block_number = 12160000
end_block_number = 4200000
step_len = -20000
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

Ethereum Blockchain source: [Infura](https://infura.io/)

Tool: [Web3 API](https://web3py.readthedocs.io/en/stable/)
