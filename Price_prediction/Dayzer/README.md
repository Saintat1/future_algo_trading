## Dayzer Prediction Evaluation
#### Dayzer Predcition Error (2018-08-24 to 2019-05-15)
![alt text](ResultOutput/dazyer_error_asof_7AM.png)

* extreme value : 2019-01-23
* dayzer was adjusted on: 
    * 2019-03-17
    * 2019-04-11
    * 2019-05-13
#### Boxplot of hourly prediction change
![alt text](ResultOutput/dayzer_price_change_boxplot_peak.png)

![alt text](ResultOutput/dayzer_price_change_boxplot_offpeak.png)
* CI for box: 5% ~ 95%
* Change on peak hour is more volatile, as expected

#### Histogram of error (DALMP - DALMP_PRED)

![alt text](ResultOutput/dayzer_error_hist_peak.png)

![alt text](ResultOutput/dayzer_error_hist_offpeak.png)

* For both peak and off-peak, Dayzer tends to **under-estimate** the DALMP
* The "closer" prediction is more accurate

#### Residual Analysis

Regression Actual ~ Prediction 

##### regression summary
Peak
![alt text](ResultOutput/lm_summary_peak.PNG)

Off-peak
![alt text](ResultOutput/lm_summary_offpeak.PNG)


##### QQ-Plot

Peak 
![alt text](ResultOutput/norm_peak_qqplot.png)

Off-peak
![alt text](ResultOutput/norm_offpeak_qqplot.png)

* None of them follow normal distribution

##### Residual ~ Fitted Value

![alt text](ResultOutput/res_fitted_peak.png)

![alt text](ResultOutput/res_fitted_offpeak.png)


