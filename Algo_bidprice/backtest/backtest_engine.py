import pandas as pd
import numpy as np

import API.productInfo as products
import Algo_bidprice.backtest.backtestUtil as btu
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

#import importlib
#importlib.reload(btu)

product = products.prodInfo("PDA")
'''
DayzerDaily
DayzerDaily_adjusted
DayzerHourly
DayzerHourly_adjusted
alllong
allshort
truevalue
LH
linearensemble
'''
result = btu.backTest(product=product,
                      margin=0,
                      maxLongVolume=np.Inf,
                      maxShortVolume=-np.Inf,
                      dateStart="2018-09-01",
                      dateEnd="2019-03-01",
                      method="allshort")
                      #method="DayzerDaily_adjusted")

performanceSummary,winrate = btu.performance(product, result)
performanceSummary = performanceSummary.fillna(0)\
    .assign(cumPnL_long = lambda df: df.PnL_long.cumsum(),
            cumPnL_short = lambda df: df.PnL_short.cumsum(),
            PnLMWh = lambda df: (df.PnL_long+df.PnL_short)/(abs(df.volume_long) + abs(df.volume_short)))





'''
PnL / #Deals / speculation volume by flow date
'''
import datetime
#performanceSummary.flowDate = pd.to_datetime(performanceSummary.flowDate)

plt.rc('xtick',labelsize = 5.5)

ax = plt.subplot(4, 1, 1)


plt.plot(performanceSummary.flowDate,
     performanceSummary.cumPnL_short,
     color='red',
     linewidth = 1,
     label="Short PnL")
plt.plot(performanceSummary.flowDate,
     performanceSummary.cumPnL_long,
     color = 'green',
     linewidth = 1,
     label="Long PnL")
plt.plot(performanceSummary.flowDate,
     performanceSummary.cumPnL_long + performanceSummary.cumPnL_short,
     color = 'blue',
    linestyle='dashed',
     linewidth = 1,
     label="Total PnL")

ymajorLocator = MultipleLocator(500000)
yminorLocator = MultipleLocator(100000)

# plt.xticks(range(len(performanceSummary.flowDate))[::5], performanceSummary.flowDate[::5])

ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
ax.yaxis.grid(True, which='major')
plt.grid(True)
plt.legend()

plt.ylabel('PnL($)')
plt.title('PnL Plot (0, Inf, -Inf) \n Long Win Rate = %1.3f%% \n Short Win Rate = %1.3f%%' %(winrate['long']*100,winrate['short']*100))
plt.xticks(rotation=90)

plt.subplot(4, 1, 2)
plt.bar(performanceSummary.flowDate,
        performanceSummary.count_long,
        color='green', align='center',label="long transactions",
        alpha=0.5)
plt.bar(performanceSummary.flowDate,
        -performanceSummary.count_short,
        color='red', align='center',label="short transactions",
        alpha=0.5)
# plt.xticks(range(len(performanceSummary.flowDate))[::5], performanceSummary.flowDate[::5])
plt.xticks(rotation=90)
plt.ylabel('Number of Deals')

plt.subplot(4, 1, 3)
plt.bar(performanceSummary.flowDate,
        performanceSummary.volume_long,
        color='green', align='center',label="long volume",
        alpha=0.5)
plt.bar(performanceSummary.flowDate,
        performanceSummary.volume_short,
        color='red', align='center',label="short volume",
        alpha=0.5)
# plt.xticks(range(len(performanceSummary.flowDate))[::5], performanceSummary.flowDate[::5])
plt.xticks(rotation=90)
plt.ylabel('Volume')

plt.subplot(4, 1, 4)
plt.bar(performanceSummary.flowDate, performanceSummary.PnLMWh, align='center', alpha=0.5,color = 'green')
# plt.xticks(range(len(performanceSummary.flowDate))[::5], performanceSummary.flowDate[::5])
plt.xticks(rotation=90)
plt.ylabel('PnL per MWh')
plt.xlabel('Date')

plt.show()
plt.subplots_adjust(hspace=0.5)
plt.close()

'''
PnL heatmap
'''
import seaborn as sns
pnlTable = pd.DataFrame(columns=[str(x) for x in range(25)],
                        index=[str(x) for x in range(7)])

for margin in range(7):
    for unit in range(25):
        result = btu.backTest(product=product,
                              margin=margin,
                              maxLongVolume=800 * unit,
                              maxShortVolume=-800 * unit / 5,
                              datePivot="2019-03-20",
                              method="dayzer")
        if result.shape[0] == 0:
            pnlTable.at[str(margin), str(unit)] = 0
        else:
            performanceSummary = btu.performance(product, result)
            PnL = performanceSummary.PnL.sum()
            pnlTable.at[str(margin), str(unit)] = PnL
        print(pnlTable)

pnlTable.to_csv('Algo_bidprice/backtest/backtest_result/' + product.productCode + '/pnlTable.csv')
ax = sns.heatmap(pnlTable.values.astype(float),
                 cmap="YlGnBu")
ax.set(xlabel='Volume Limitation(1 unit = 800 MWs)',
       ylabel='Margin($)',
       title="PnL Heatmap (Long/short = 5)")
plt.show()
