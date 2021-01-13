# ChainNet

## 1. Time series of daily log returns, transactions, average β0 and β1 numbers in 2017.
   
[Source File](./paper_result/experiment_data/fig2.ipynb)

![image](./paper_result/experiment_data/data_2017_total_tx_log.jpg)
![image](./paper_result/experiment_data/data_2017_total_tx.jpg)
![image](./paper_result/experiment_data/betti_0.jpg)
![image](./paper_result/experiment_data/betti_1.jpg)



## 2. RMSE of sliding window based predictions of 2017 Bitcoin prices in different window and horizon values.

Models: RF, ENET, GP, XGBT

[Source File](./paper_result/best_performance/best_performance.py)

<b>window_size = 3</b>

![image](./paper_result/best_performance/result_image/base_rmse_window_3_line.png)

<b>window_size = 5</b>

![image](./paper_result/best_performance/result_image/base_rmse_window_5_line.png)

<b>window_size = 7</b>

![image](./paper_result/best_performance/result_image/base_rmse_window_7_line.png)


Models: RF, ENET, GP, XGBT, ARIMAX (last version)

[Source File](./paper_result/experiment_rmse/rmse_models.py)

<b>window_size = 3</b>

![image](./paper_result/experiment_rmse/version_12_14_09_10/WINDOW_3_5.png)

<b>window_size = 5</b>

![image](./paper_result/experiment_rmse/version_12_14_09_10/WINDOW_5_5.png)

<b>window_size = 7</b>

![image](./paper_result/experiment_rmse/version_12_14_09_10/WINDOW_7_5.png)



## 3. [Source File](./paper_result/best_performance/best_performance.py)

    3.1 Random Forest Performance.

![image](./paper_result/best_performance/result_image/rf_window_3.png)
![image](./paper_result/best_performance/result_image/rf_window_5.png)
![image](./paper_result/best_performance/result_image/rf_window_7.png)

    3.2 Elastic Net model performance.

![image](./paper_result/best_performance/result_image/enet_window_3.png)
![image](./paper_result/best_performance/result_image/enet_window_5.png)
![image](./paper_result/best_performance/result_image/enet_window_7.png)
    
    3.3 GP performance:

![image](./paper_result/best_performance/result_image/gp_window_3.png)
![image](./paper_result/best_performance/result_image/gp_window_5.png)
![image](./paper_result/best_performance/result_image/gp_window_7.png)

    3.4 XGBT performance:

![image](./paper_result/best_performance/result_image/xgbt_window_3.png)
![image](./paper_result/best_performance/result_image/xgbt_window_5.png)
![image](./paper_result/best_performance/result_image/xgbt_window_7.png)

[Result data](./paper_result/best_performance/result_data/)

## Reference:

[ChainNet Paper](https://arxiv.org/pdf/1908.06971)
