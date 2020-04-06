
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import importlib
import pandas as pd
import API.dalmp_Dayzer as dalmp
import API.utilFunction as util
import API.productInfo as products
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.graphics.gofplots import ProbPlot

#importlib.reload(dalmp)

product = products.prodInfo("PDA")
startdate = "2019-03-20"
enddate = "2019-05-13"

'''
Accuracy:
MAE,MAPE,MSE
'''
realDalmp = dalmp.dalmp_real(product.iso(),
                             product.nodeid(),
                              "dalmp",
                             startdate,
                             enddate,
                             hourly=True)

realDalmpAgg = dalmp.dalmp_real(product.iso(),
                             product.nodeid(),
                              "dalmp",
                             startdate,
                             enddate,
                             hourly=False)


realDalmp = pd.melt(realDalmp,
                    id_vars='flow_date',
                    value_vars=["he" + str(i) for i in range(1, 25)],
                    value_name="DALMP")

realDalmp = realDalmp.rename(columns={'variable': "hour", }) \
    .assign(hour=lambda df: df.hour.str[2:].astype(int)) \
    .sort_values(by=["flow_date", "hour"])

predictDalmp = dalmp.dalmp_pred("PJMISO", "51288", "dalmp", startdate, enddate, hourly=True)

predictDalmp = predictDalmp.dropna() \
    .sort_values(by=["flow_date", "update_time"]) \
    .groupby("flow_date") \
    .tail(1) \
    .melt(id_vars=['flow_date'], value_vars=["he" + str(i) for i in range(1, 25)], value_name="DALMP_predicted") \
    .rename(columns={'variable': "hour"}) \
    .assign(hour=lambda df: df.hour.str[2:].astype(int)) \
    .sort_values(by=["flow_date", "hour"])

finalDF = realDalmp.merge(predictDalmp, how="left", on=["flow_date", "hour"]) \
    .dropna() \
    .assign(APE=lambda df: abs(df.DALMP - df.DALMP_predicted) / df.DALMP,
            SE=lambda df: (df.DALMP - df.DALMP_predicted) ** 2,
            AE=lambda df: abs(df.DALMP - df.DALMP_predicted))

MAPE = finalDF.APE.mean()
MSE = finalDF.SE.mean()
MAE = finalDF.AE.max()

'''
Dayzer trend
'''

predictDalmpAgg = dalmp.dalmp_pred("PJMISO", "51288", "dalmp", startdate, enddate, hourly=False) \
    .dropna() \
    .sort_values(by=["flow_date","update_time"]) \
    .merge(realDalmpAgg, how = "left", on = ["flow_date","node_id"])

#predictDalmpAgg["onpeak_pred_shift"] = predictDalmpAgg.groupby(['flow_date'])['onpeak_pred'] \
 #   .shift(1)
#predictDalmpAgg["offpeak_pred_shift"] = predictDalmpAgg.groupby(['flow_date'])['offpeak_pred'] \
 #   .shift(1)

plot_df = predictDalmpAgg.assign(#onpeak_pred_change = lambda df: df.onpeak_pred - df.onpeak_pred_shift,
                                 #offpeak_pred_change = lambda  df : df.offpeak_pred - df.offpeak_pred_shift,
                                 #onpeak_pred_change_abs = lambda df: abs(df.onpeak_pred - df.onpeak_pred_shift),
                                 #offpeak_pred_change_abs = lambda  df : (df.offpeak_pred - df.offpeak_pred_shift),
                                 error_peak =  lambda  df : (df.onpeak_avg - df.onpeak_pred),
                                 error_offpeak =  lambda  df : (df.offpeak_avg- df.offpeak_pred),
                                 abs_error_peak =  lambda  df : abs(df.onpeak_avg - df.onpeak_pred),
                                 abs_error_offpeak =  lambda  df : abs(df.offpeak_avg- df.offpeak_pred),
                                 hour = lambda df: pd.to_datetime(df.update_time).dt.hour,
                                 update_date = lambda df: pd.to_datetime(df.update_time).dt.date) \
    .pipe(util.select,["flow_date",
                       "update_time",
                       "onpeak_avg",
                       "offpeak_avg",
                       "onpeak_pred",
                       "offpeak_pred",
                       #"onpeak_pred_change",
                       #"offpeak_pred_change",
                       #"onpeak_pred_change_abs",
                       #"offpeak_pred_change_abs",
                       "error_peak",
                       "error_offpeak",
                       "abs_error_offpeak",
                       "abs_error_peak",
                       "abs_error_offpeak",
                       "hour",
                       "update_date"])\
    .dropna()\
    .sort_values(['flow_date','update_time']).groupby('flow_date').tail(11)
########
# fc plot
#########
plot_df = plot_df.query('hour >=8')

plt.plot('update_time','onpeak_pred',data=plot_df,marker='o', markerfacecolor='blue', markersize=1, color='skyblue', linewidth=1)
plt.plot( 'flow_date', 'onpeak_avg', data=plot_df, marker='', color='olive', linewidth=2)
plt.legend()

plt.scatter('update_time','onpeak_pred',data =plot_df , s=75, c="r", alpha=.5)
plt.plot( 'update_time', 'onpeak_pred', data=plot_df, linestyle='none', marker='o',markersize=0.5,color='forecast_hour', hue="flow_date")

from ggplot import *
ggplot(aes(x='update_time', y='onpeak_pred', color='forecast_hour'), data=plot_df) +\
geom_point(size=1) +\
theme_bw()

sns.pairplot(x_vars=["update_time"], y_vars=["onpeak_pred"], data=plot_df, hue="flow_date", size=5)

###############
# peak delta
###############
plot_df.pivot(columns = 'flow_date', values = ['onpeak_pred_change']) \
    .apply(lambda x: pd.Series(x.dropna().values)) \
    .boxplot(rot=90)

#plt.rc('xtick',labelsize = 10)
plt.title("Dayzer Prediction Change(Peak)")
plt.xlabel("Date")
plt.ylabel("Price change")

###############
# Off-peak delta
###############
plot_df.pivot(columns = 'flow_date', values = ['offpeak_pred_change']) \
    .apply(lambda x: pd.Series(x.dropna().values)) \
    .boxplot(rot=90)


#plt.rc('xtick',labelsize = 10)
plt.title("Dayzer Prediction Change(Off-Peak)")
plt.xlabel("Date")
plt.ylabel("Price change")
plt.show()


plt.show()

#####
#Accuracy hist
#####
bins = np.arange(-9,9,1)

g = sns.FacetGrid(plot_df,col="hour")
g.map(plt.hist,"error_peak",bins=bins)
plt.show()

bins = np.arange(-5,5,1)
g = sns.FacetGrid(plot_df,col="hour")
g.map(plt.hist,"error_offpeak",bins=bins)

#################
# Accuracy regression & QQ Plot
#################

startdate = "2018-08-24"
enddate = "2019-05-22"

predictDalmpDaily = dalmp.dalmp_pred_vintage(iso = product.iso(),
                                            node_id= product.nodeid(),
                                             data_type= "dalmp",
                                             start_date= startdate,
                                             end_date= enddate)

realDalmpAgg = dalmp.dalmp_real(product.iso(),
                             product.nodeid(),
                              "dalmp",
                             startdate,
                             enddate,
                             hourly=False)

regressionDF = realDalmpAgg.merge(predictDalmpDaily, how = 'left', on = ['flow_date','node_id']) \
    .dropna() \
    .assign(peak_error = lambda df: df.onpeak_avg - df.onpeak_pred,
            offpeak_error = lambda  df : df.offpeak_avg - df.offpeak_pred)

plt.plot( 'flow_date', 'peak_error', data=regressionDF, marker='', color='blue', linewidth=0.5)
plt.plot( 'flow_date', 'offpeak_error', data=regressionDF, marker='', color='red', linewidth=1)
plt.axhline(y=0, color='black', linestyle='--', linewidth = 0.5)
#plt.axvline(x="2019-01-23",color='r', linestyle='--',linewidth = 0.5)
#plt.plot( 'x', 'y3', data=regressionDF, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
plt.legend()

Y = regressionDF.onpeak_avg
X = regressionDF.onpeak_pred

mod_fit_peak = sm.OLS(Y,X).fit()

mod_fit_peak_y = mod_fit_peak.fittedvalues

mod_fit_peak_res = mod_fit_peak.resid

mod_fit_peak_normres = mod_fit_peak.get_influence().resid_studentized_internal

mod_fit_peak_normres_abs_sqrt = np.sqrt(np.abs(mod_fit_peak_normres))

mod_fit_peak_abs_res = np.abs(mod_fit_peak_res)

mod_fit_peak_leverage = mod_fit_peak.get_influence().hat_matrix_diag

plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

plot_lm_1.axes[0] = sns.residplot(mod_fit_peak_y, 'onpeak_avg', data=regressionDF,
                          lowess=True,
                          scatter_kws={'alpha': 0.5},
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted(Peak)')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')

# annotations
abs_resid = mod_fit_peak_abs_res.sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot_lm_1.axes[0].annotate(i,
                               xy=(mod_fit_peak_y[i],
                                   mod_fit_peak_res[i]));


QQ = ProbPlot(mod_fit_peak_normres)
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

plot_lm_2.set_figheight(8)
plot_lm_2.set_figwidth(12)

plot_lm_2.axes[0].set_title('Normal Q-Q(Peak)')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

# annotations
abs_norm_resid = np.flip(np.argsort(np.abs(mod_fit_peak_normres)), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]

for r, i in enumerate(abs_norm_resid_top_3):
    plot_lm_2.axes[0].annotate(i,
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   mod_fit_peak_normres[i]));


####################offpeak###################################

mod_fit_offpeak = sm.OLS(regressionDF.offpeak_avg, regressionDF.offpeak_pred).fit()

mod_fit_offpeak_y = mod_fit_offpeak.fittedvalues

mod_fit_offpeak_res = mod_fit_offpeak.resid

mod_fit_offpeak_normres = mod_fit_offpeak.get_influence().resid_studentized_internal

mod_fit_offpeak_normres_abs_sqrt = np.sqrt(np.abs(mod_fit_offpeak_normres))

mod_fit_offpeak_abs_res = np.abs(mod_fit_offpeak_res)

mod_fit_offpeak_leverage = mod_fit_offpeak.get_influence().hat_matrix_diag

plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

plot_lm_1.axes[0] = sns.residplot(mod_fit_offpeak_y, 'offpeak_avg', data=regressionDF,
                          lowess=True,
                          scatter_kws={'alpha': 0.5},
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted(Off-Peak)')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')

# annotations
abs_resid = mod_fit_offpeak_abs_res.sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot_lm_1.axes[0].annotate(i,
                               xy=(mod_fit_offpeak_y[i],
                                   mod_fit_offpeak_res[i]));


QQ = ProbPlot(mod_fit_offpeak_normres)
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

plot_lm_2.set_figheight(8)
plot_lm_2.set_figwidth(12)

plot_lm_2.axes[0].set_title('Normal Q-Q(Off-Peak)')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

# annotations
abs_norm_resid = np.flip(np.argsort(np.abs(mod_fit_offpeak_normres)), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]

for r, i in enumerate(abs_norm_resid_top_3):
    plot_lm_2.axes[0].annotate(i,
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   mod_fit_offpeak_normres[i]));

