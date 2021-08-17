# ETH ChainNet

## 1. Bitcoin and ETH Occurrence matrix.

[Data Preparation](./processed_data_07_31___08_29/final_process_step_2.py)

![image](../Figures/BitcoinOccMatrix.png)

![image](../Figures/EthereumOccMatrix.png)
   

## 2. RMSE of sliding window based predictions of Ethereum prices in different window and horizon values. 

### 2.1. YEAR = 2020

Models: RF, ENET, GP, XGBT

[Source File](./best_performance/best_performance.py)

<b>window_size = 3: </b>

![image](./processed_data_07_31___08_29/base_rmse_window_3_line.png)

<b>window_size = 5: </b>

![image](./processed_data_07_31___08_29/base_rmse_window_5_line.png)

<b>window_size = 7: </b>

![image](./processed_data_07_31___08_29/base_rmse_window_7_line.png)

## 3. Best performance 

[Source File](./best_performance/best_performance.py)

### 3.1. YEAR = 2020

    3.1.1. Random Forest Performance.

![image](./processed_data_07_31___08_29/rf_window_3.png)
![image](./processed_data_07_31___08_29/rf_window_5.png)
![image](./processed_data_07_31___08_29/rf_window_7.png)

    3.1.2. Elastic Net model performance.

![image](./processed_data_07_31___08_29/enet_window_3.png)
![image](./processed_data_07_31___08_29/enet_window_5.png)
![image](./processed_data_07_31___08_29/enet_window_7.png)
    
    3.1.3. GP performance:

![image](./processed_data_07_31___08_29/gp_window_3.png)
![image](./processed_data_07_31___08_29/gp_window_5.png)
![image](./processed_data_07_31___08_29/gp_window_7.png)

    3.1.4. XGBT performance:

![image](./processed_data_07_31___08_29/xgbt_window_3.png)
![image](./processed_data_07_31___08_29/xgbt_window_5.png)
![image](./processed_data_07_31___08_29/xgbt_window_7.png)

[Result data](./processed_data_07_31___08_29/)


## Reference:

[ChainNet Paper](https://arxiv.org/pdf/1908.06971)
