## [TO BE CONTINUED]

## 0. ChainLet:

   (source file)[./chainlet/]

   (data)[./data/]

## 1. Time series of daily log returns, transactions, average β0 and β1 numbers in 2017.
   
![image](./paper_result/experiment_data/data_2017_total_tx_log.jpg)
![image](./paper_result/experiment_data/data_2017_total_tx.jpg)
![image](./paper_result/experiment_data/betti_0.jpg)
![image](./paper_result/experiment_data/betti_1.jpg)

[Source File](./paper_result/experiment_data/fig2.ipynb)

## 2. RMSE of sliding window based predictions of 2017 Bitcoin prices in different window and horizon values.

Models: RF, ENET, GP, XGBT

![image](./paper_result/experiment_rmse/version_12_14_09_10/WINDOW_3_4.png)
![image](./paper_result/experiment_rmse/version_12_14_09_10/WINDOW_5_4.png)
![image](./paper_result/experiment_rmse/version_12_14_09_10/WINDOW_7_4.png)

Models: RF, ENET, GP, XGBT, ARIMAX

![image](./paper_result/experiment_rmse/version_12_14_09_10/WINDOW_3_5.png)
![image](./paper_result/experiment_rmse/version_12_14_09_10/WINDOW_5_5.png)
![image](./paper_result/experiment_rmse/version_12_14_09_10/WINDOW_7_5.png)

[Source File](./paper_result/experiment_rmse/rmse_models.py)

## 3. [Source File](./paper_result/models_performance/models_performance.py)

    3.1 Random Forest Performance.

![image](./paper_result/models_performance/performance/performance_fixed_parameter/rf_window_3.png)
![image](./paper_result/models_performance/performance/performance_fixed_parameter/rf_window_5.png)
![image](./paper_result/models_performance/performance/performance_fixed_parameter/rf_window_7.png)

    3.2 Elastic Net model performance.

![image](./paper_result/models_performance/performance/performance_fixed_parameter/enet_window_3.png)
![image](./paper_result/models_performance/performance/performance_fixed_parameter/enet_window_5.png)
![image](./paper_result/models_performance/performance/performance_fixed_parameter/enet_window_7.png)
    
    3.3 GP performance:

![image](./paper_result/models_performance/performance/performance_fixed_parameter/gp_window_3.png)
![image](./paper_result/models_performance/performance/performance_fixed_parameter/gp_window_5.png)
![image](./paper_result/models_performance/performance/performance_fixed_parameter/gp_window_7.png)

    3.4 XGBT performance:

![image](./paper_result/models_performance/performance/performance_fixed_parameter/xgbt_window_3.png)
![image](./paper_result/models_performance/performance/performance_fixed_parameter/xgbt_window_5.png)
![image](./paper_result/models_performance/performance/performance_fixed_parameter/xgbt_window_7.png)



## Reference:

[ChainNet Paper](https://arxiv.org/pdf/1908.06971)