# ChainNet

## 1. Time series of daily log returns, transactions, average β0 and β1 numbers in 2017.
   
[Source File](./experiment_data/fig2.ipynb)

![image](./experiment_data/data_2017_total_tx_log.jpg)
![image](./experiment_data/data_2017_total_tx.jpg)
![image](./experiment_data/betti_0.jpg)
![image](./experiment_data/betti_1.jpg)



## 2. RMSE of sliding window based predictions of Bitcoin prices in different window and horizon values. 

### 2.1. YEAR = 2017

Models: RF, ENET, GP, XGBT

[Source File](./best_performance/best_performance.py)

<b>window_size = 3: </b>

![image](./best_performance/result_image/2017/base_rmse_window_3_line.png)

<b>window_size = 5: </b>

![image](./best_performance/result_image/2017/base_rmse_window_5_line.png)

<b>window_size = 7: </b>

![image](./best_performance/result_image/2017/base_rmse_window_7_line.png)

### 2.1. YEAR = 2020

Models: RF, ENET, GP, XGBT

[Source File](./best_performance/best_performance.py)

<b>window_size = 3: </b>

![image](./best_performance/result_image/2020/base_rmse_window_3_line.png)

<b>window_size = 5: </b>

![image](./best_performance/result_image/2020/base_rmse_window_5_line.png)

<b>window_size = 7: </b>

![image](./best_performance/result_image/2020/base_rmse_window_7_line.png)
<!-- Models: RF, ENET, GP, XGBT, ARIMAX (last version)

[Source File](./experiment_rmse/rmse_models.py)

<b>window_size = 3</b>

![image](./experiment_rmse/version_12_14_09_10/WINDOW_3_5.png)

<b>window_size = 5</b>

![image](./experiment_rmse/version_12_14_09_10/WINDOW_5_5.png)

<b>window_size = 7</b>

![image](./experiment_rmse/version_12_14_09_10/WINDOW_7_5.png) -->



## 3. Best performance 

[Source File](./best_performance/best_performance.py)

### 3.1. YEAR = 2017

    3.1.1. Random Forest Performance.

![image](./best_performance/result_image/2017/rf_window_3.png)
![image](./best_performance/result_image/2017/rf_window_5.png)
![image](./best_performance/result_image/2017/rf_window_7.png)

    3.1.2. Elastic Net model performance.

![image](./best_performance/result_image/2017/enet_window_3.png)
![image](./best_performance/result_image/2017/enet_window_5.png)
![image](./best_performance/result_image/2017/enet_window_7.png)
    
    3.1.3. GP performance:

![image](./best_performance/result_image/2017/gp_window_3.png)
![image](./best_performance/result_image/2017/gp_window_5.png)
![image](./best_performance/result_image/2017/gp_window_7.png)

    3.1.4. XGBT performance:

![image](./best_performance/result_image/2017/xgbt_window_3.png)
![image](./best_performance/result_image/2017/xgbt_window_5.png)
![image](./best_performance/result_image/2017/xgbt_window_7.png)

[Result data](./best_performance/result_data/2017/)

### 3.2. YEAR = 2020

    3.2.1. Random Forest Performance.

![image](./best_performance/result_image/2020/rf_window_3.png)
![image](./best_performance/result_image/2020/rf_window_5.png)
![image](./best_performance/result_image/2020/rf_window_7.png)

    3.2.2. Elastic Net model performance.

![image](./best_performance/result_image/2020/enet_window_3.png)
![image](./best_performance/result_image/2020/enet_window_5.png)
![image](./best_performance/result_image/2020/enet_window_7.png)
    
    3.2.3. GP performance:

![image](./best_performance/result_image/2020/gp_window_3.png)
![image](./best_performance/result_image/2020/gp_window_5.png)
![image](./best_performance/result_image/2020/gp_window_7.png)

    3.2.4. XGBT performance:

![image](./best_performance/result_image/2020/xgbt_window_3.png)
![image](./best_performance/result_image/2020/xgbt_window_5.png)
![image](./best_performance/result_image/2020/xgbt_window_7.png)

[Result data](./best_performance/result_data/2020/)

## Reference:

[ChainNet Paper](https://arxiv.org/pdf/1908.06971)
