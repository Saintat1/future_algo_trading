import pandas as pd
#import seaborn as sns
import API.productInfo as products
import API.orderBookAPI.orderBookApi as ob
from pylab import *
import API.utilFunction as util
import API.dalmp_Dayzer as dalmp_dayzer
from datetime import timedelta, date
#import importlib
#importlib.reload(ob)
import BackTest.backtest.backtestUtil as btu


product = products.prodInfo("PDA")
dateStart = "2018-10-01"
dateEnd = "2019-02-28"

datePivotStart = datetime.datetime.strptime(dateStart, "%Y-%m-%d")
datePivotEnd = datetime.datetime.strptime(dateEnd, "%Y-%m-%d")

tick_df1 = pd.read_excel("./Algo_bidprice/Data/2019 power deals.xlsx")\
    .pipe(util.select, ['STRIP','HUB','STRIP BEGIN', 'ORDER PRICE', 'DEAL_TIME', 'LOTS', 'TOTAL VOLUME', 'BID/OFFER', 'IS BLOCK'])

tick_df2 = pd.read_excel("./Algo_bidprice/Data/Guzman Deal Data.xlsx") \
    .pipe(util.select, ['STRIP','HUB','STRIP BEGIN', 'ORDER PRICE', 'DEAL_TIME', 'LOTS', 'TOTAL VOLUME', 'BID/OFFER', 'IS BLOCK'])

tick_df = pd.concat([tick_df1, tick_df2], sort=False) \
    .drop_duplicates()

tick_df = tick_df.query('STRIP == @product.strip()') \
    .query('HUB == @product.hub()') \
    .pipe(util.select, ['STRIP BEGIN', 'ORDER PRICE', 'DEAL_TIME', 'LOTS', 'TOTAL VOLUME', 'BID/OFFER', 'IS BLOCK']) \
    .rename(columns={'STRIP BEGIN': 'strip_begin',
                     'ORDER PRICE': 'order_price',
                     'DEAL_TIME': 'deal_time',
                     'LOTS': 'lots',
                     'TOTAL VOLUME': 'volume',
                     'BID/OFFER': 'type',
                     'IS BLOCK': 'is_block'
                     }) \
    .query('is_block != "Y"') \
    .assign(flow_date=lambda df: pd.to_datetime(df.strip_begin).dt.to_pydatetime(),
            deal_time=lambda df: pd.to_datetime(df.deal_time).dt.to_pydatetime()) \
    .pipe(util.select,['flow_date','deal_time','order_price','volume','type']) \
    .query('flow_date >= @datePivotStart') \
    .query('flow_date <= @datePivotEnd') \
    .sort_values('deal_time')\
    .assign(flow_date=lambda df: df.flow_date.dt.strftime('%Y-%m-%d'))

peakType = "onpeak_avg" if product.peaktype() == "onpeak" else "offpeak_avg"

histDalmp = dalmp_dayzer.dalmp_real(product.iso(),
                                        product.nodeid(),
                                        "dalmp",
                                        dateStart,
                                        dateEnd)[["flow_date", peakType]] \
        .rename(columns={peakType: 'settleDalmp',
                         'flow_date': 'flowDate'})

histDalmp.flowDate = histDalmp.flowDate.astype(str)


predict_dalmp = btu.getDalmpPred("DayzerDaily_adjusted", "", product, dateStart)

iso = product.iso()
node_id = product.nodeid()
data_type = "dalmp"
peak_type = product.peaktype()

peak_adjust_value = 0.8795
offpeak_adjust_value = 0.9036

predict_dalmp = dalmp_dayzer.dalmp_pred_vintage(iso=iso,
                                            node_id=node_id,
                                            data_type=data_type,
                                            start_date=dateStart,
                                            end_date=dateEnd)
dalmpPred = predict_dalmp.assign(onpeak_pred_adjusted = lambda df: df.onpeak_pred* peak_adjust_value,
                                 offpeak_pred_adjusted = lambda df: df.offpeak_pred* offpeak_adjust_value)

dalmpPred = predict_dalmp
dalmpPred.flow_date = dalmpPred.flow_date.astype(str)

plot_df = tick_df.merge(histDalmp,how='left',left_on='flow_date',right_on = 'flowDate') \
    .merge(dalmpPred,how='left',on='flow_date')\
    .pipe(util.select,['deal_time','type','order_price','settleDalmp','onpeak_pred','onpeak_pred_adjusted'])\
    .dropna()
'''
plot_df = tick_df.merge(histDalmp,how='left',left_on='flow_date',right_on = 'flowDate') \
    .pipe(util.select,['deal_time','type','order_price','settleDalmp','flow_date'])\
    .assign(delta = lambda df: df.settleDalmp - df.order_price)


import matplotlib.pyplot as plt
import seaborn as sns;
bins = np.arange(-15,15,0.5)
g = sns.FacetGrid(plot_df,col="type")
g = g.map(plt.hist, "delta",bins=bins)
plt.close()
'''

fig, ax = plt.subplots()

for key, grp in plot_df.groupby(['type']):
    ax = grp.plot(ax=ax,
                  x='deal_time',
                  y='order_price',
                  #c= 'red' if key =='Bid' else 'green',
                  linestyle='',
                  marker='o',
                  ms=3,
                  label=key)

ax = plot_df.plot(ax=ax,
              x='deal_time',
              y='settleDalmp',
              #c= 'red' if key =='Bid' else 'green',
              linestyle='-',
              label="Settle Price")

ax = plot_df.plot(ax=ax,
              x='deal_time',
              y='onpeak_pred',
              #c= 'red' if key =='Bid' else 'green',
              linestyle='--',
              label="prediction")

ax = plot_df.plot(ax=ax,
              x='deal_time',
              y='onpeak_pred_adjusted',
              #c= 'red' if key =='Bid' else 'green',
              linestyle='--',
              label="prediction adjusted")


plt.legend(loc='best')
plt.show()
plt.title('Flow_date: %s' %dateStart)
#plt.close()