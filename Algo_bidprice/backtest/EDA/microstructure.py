import pandas as pd
#import seaborn as sns
import API.productInfo as products
import API.orderBookAPI.orderBookApi as ob
from pylab import *
import API.utilFunction as util
from datetime import timedelta, date
#import importlib
#importlib.reload(ob)

product = products.prodInfo("PDA")

orderbook = ob.OrderBook(product)

orderbook.reset()
orderbook.dailybook("2019-01-07")

orderbook.bestbid
orderbook.bestask
orderbook.askstack
orderbook.bidstack
orderbook.trade

'''
Bid/Ask plot
'''
askline = orderbook.askstack.pipe(util.select,["price","volume"]).assign(type = 'ask').sort_values("price",ascending=True)
askline["cumvolume"] = askline.volume.cumsum()

bidline = orderbook.bidstack.pipe(util.select,["price","volume"]).assign(type = 'bid').sort_values("price",ascending=False)
bidline["cumvolume"] = bidline.volume.cumsum()

askline.append(bidline).pivot(index = 'price',columns = 'type',values = 'cumvolume').plot()

plt.show()

'''
trade ratio
'''
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

start_date = date(2019, 1, 2)
end_date = date(2019, 4, 23)

summaryTable = pd.DataFrame(columns=['Trade','Quotes','Ratio'],
                        index=[single_date.strftime("%Y-%m-%d") for single_date in daterange(start_date, end_date)])


for index, row in summaryTable.iterrows():
    print(index)
    orderbook.reset()
    orderbook.dailybook(index)
    summaryTable.at[index,"Trade"] = orderbook.trade.shape[0]
    summaryTable.at[index, "Quotes"] = orderbook.trade.shape[0]*2 + orderbook.askstack.shape[0] + orderbook.bidstack.shape[0]
    summaryTable.at[index, "Ratio"] = None if summaryTable.at[index, "Quotes"] == 0 else summaryTable.at[index,"Trade"]/summaryTable.at[index, "Quotes"]

summaryTable = summaryTable.query('Quotes != 0')

summaryTable.pipe(util.select,['Trade']).plot.hist()
summaryTable.pipe(util.select,['Quotes']).plot.hist()
summaryTable.pipe(util.select,['Ratio']).plot.hist()