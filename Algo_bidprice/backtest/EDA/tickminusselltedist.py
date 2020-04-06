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
dateStart = "2018-03-01"
dateEnd = "2019-03-01"

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




plot_df = tick_df.merge(histDalmp,how='left',left_on='flow_date',right_on = 'flowDate') \
    .pipe(util.select,['deal_time','type','order_price','settleDalmp','flow_date'])\
    .assign(delta = lambda df: df.settleDalmp - df.order_price)


import matplotlib.pyplot as plt
import seaborn as sns;
bins = np.arange(-15,15,0.5)
g = sns.FacetGrid(plot_df,col="type")
g = g.map(plt.hist, "delta",bins=bins)
plt.title('From %s to %s' %(dateStart,dateEnd))
plt.close()
