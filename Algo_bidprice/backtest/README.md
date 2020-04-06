## BackTest Result (PDA)

#### Parameter
1. **Threshold**
   * Actual bid price = dalmp(predicted) - margin
   * Actual offer price =  dalmp(predicted) + margin

2. **Long short ratio**: defined as (max long volume) / (max short volume)
3. **Volume limitation**
4. **Prediction method**
    * Dayzer hourly
    * Dayzer daily
    * Dayzer daily adjusted
    * Like Hour(TODO)
---
### Extreme Situation

#### All short
![alt text](backtest_result/PDA/allshort.png)

#### All long
![alt text](backtest_result/PDA/alllong.png)

#### Trade with settled DALMP
![alt text](backtest_result/PDA/godview.png)

---
### Dayzer Performance

#### Threshold = 0, (-inf,inf), Dayzer Daily Prediction
![alt text](backtest_result/PDA/pnl_performance_unlimited_0th_dayzerdaily.png)

#### Threshold = 0, (-inf,inf), Dayzer Daily Adjusted Prediction
![alt text](backtest_result/PDA/pnl_performance_unlimited_0th_dayzerdailyadjusted.png)

##### Big win detail
![alt text](backtest_result/PDA/priceplot_20190111.png)

![alt text](backtest_result/PDA/priceplot_20190121.png)

##### Big loss detail

![alt text](backtest_result/PDA/priceplot_20181204.png)

---
### Like-Hour Performance

![alt text](backtest_result/PDA/pnl_performance_unlimited_0th_lhdaily.png)

---

### Ensembling

#### Linear ensembling
![alt text](backtest_result/PDA/pnl_performance_unlimited_0th_linearensemble.png)

#### ML ensembling(TODO)

