## Generate the betti sequence

Because of the unclear documentation of the paper: [ChainNet](). So made this document to record the details of the method to generate the betti sequence based on the dynamic graph data.

1. Transfer the Satoshis with the log function: 
   
   a' = log ( 1 + a / $10 ^ 8$ )

2. For each chainlet of a given time period, we compute the sample q-quantiles for the associated log transfered amounts: a k-th q-quantile, k=0, 1, 2, ... q, is the amount Q(k) such that:
   
   