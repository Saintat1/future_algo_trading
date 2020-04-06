import datetime

import pandas as pd
import numpy as np

import API.productInfo as products
import API.dalmp_Dayzer as dalmp_dayzer
import API.dalmp_LH as dalmp_likehour
import API.utilFunction as util
import importlib
importlib.reload(dalmp_likehour)


def parseTickData(product, dateStart,dateEnd):
    datePivotStart = datetime.datetime.strptime(dateStart, "%Y-%m-%d")
    datePivotEnd = datetime.datetime.strptime(dateEnd, "%Y-%m-%d")

    tick_df1 = pd.read_excel("./Algo_bidprice/Data/2019 power deals.xlsx") \
        .pipe(util.select,
              ['STRIP', 'HUB', 'STRIP BEGIN', 'ORDER PRICE', 'DEAL_TIME', 'LOTS', 'TOTAL VOLUME', 'BID/OFFER',
               'IS BLOCK'])

    tick_df2 = pd.read_excel("./Algo_bidprice/Data/Guzman Deal Data.xlsx") \
        .pipe(util.select,
              ['STRIP', 'HUB', 'STRIP BEGIN', 'ORDER PRICE', 'DEAL_TIME', 'LOTS', 'TOTAL VOLUME', 'BID/OFFER',
               'IS BLOCK'])

    tick_df = pd.concat([tick_df1,tick_df2],sort=False) \
        .drop_duplicates()

    tick_df = tick_df.query('STRIP == @product.strip()') \
        .query('HUB == @product.hub()') \
        .pipe(util.select, ['STRIP BEGIN', 'ORDER PRICE', 'DEAL_TIME', 'LOTS', 'TOTAL VOLUME', 'BID/OFFER', 'IS BLOCK']) \
        .rename(columns={'STRIP BEGIN': 'strip_begin',
                         'ORDER PRICE': 'order_price',
                         'DEAL_TIME': 'deal_time',
                         'LOTS': 'lots',
                         'TOTAL VOLUME': 'total_volume',
                         'BID/OFFER': 'bid/offer',
                         'IS BLOCK': 'is_block'
                         }) \
        .query('is_block != "Y"') \
        .assign(strip_begin=lambda df: pd.to_datetime(df.strip_begin).dt.to_pydatetime(),
                deal_time=lambda df: pd.to_datetime(df.deal_time).dt.to_pydatetime()) \
        .query('strip_begin >= @datePivotStart') \
        .query('strip_begin <= @datePivotEnd') \
        .sort_values('deal_time')

    return tick_df


def getDalmpPred(method, timestamp, product, flow_date):
    iso = product.iso()
    node_id = product.nodeid()
    data_type = "dalmp"
    peak_type = product.peaktype()

    peak_adjust_value = 0.8795
    offpeak_adjust_value = 0.9036

    if method == 'DayzerHourly':
        dalmpPred = dalmp_dayzer.dalmp_pred_closest(iso=iso,
                                                    node_id=node_id,
                                                    data_type=data_type,
                                                    flow_date=flow_date,
                                                    deal_datetime=timestamp)

        dalmpPred = dalmpPred["onpeak_pred"] if peak_type == "onpeak" else dalmpPred[
            "offpeak_pred"]

        return dalmpPred

    elif method == 'DayzerHourly_adjusted':

        dalmpPred = dalmp_dayzer.dalmp_pred_closest(iso=iso,
                                                    node_id=node_id,
                                                    data_type=data_type,
                                                    flow_date=flow_date,
                                                    deal_datetime=timestamp)

        dalmpPred = dalmpPred["onpeak_pred"] * peak_adjust_value if peak_type == "onpeak" else dalmpPred[
                                                                                                   "offpeak_pred"] * offpeak_adjust_value
        return dalmpPred

    elif method == 'DayzerDaily':
        dalmpPred = dalmp_dayzer.dalmp_pred_vintage(iso=iso,
                                                    node_id=node_id,
                                                    data_type=data_type,
                                                    start_date=flow_date,
                                                    end_date=flow_date)
        dalmpPred = dalmpPred["onpeak_pred"] if peak_type == "onpeak" else dalmpPred["offpeak_pred"]
        return dalmpPred

    elif method == 'DayzerDaily_adjusted':
        dalmpPred = dalmp_dayzer.dalmp_pred_vintage(iso=iso,
                                                    node_id=node_id,
                                                    data_type=data_type,
                                                    start_date=flow_date,
                                                    end_date=flow_date)
        dalmpPred = dalmpPred["onpeak_pred"] * peak_adjust_value if peak_type == "onpeak" else dalmpPred[
                                                                                                  "offpeak_pred"] * offpeak_adjust_value
        return dalmpPred

    elif method == 'allshort':
        return pd.DataFrame({'onpeak_pred':[-np.Inf]}).iloc[0]

    elif method == 'alllong':
        return pd.DataFrame({'onpeak_pred':[np.Inf]}).iloc[0]

    elif method == 'truevalue':
        dalmpPred = dalmp_dayzer.dalmp_real(iso=iso,
                                                    node_id=node_id,
                                                    data_type=data_type,
                                                    start_date=flow_date,
                                                    end_date=flow_date,
                                                    hourly=False)
        dalmpPred = dalmpPred["onpeak_avg"] if peak_type == "onpeak" else dalmpPred["offpeak_avg"]
        return dalmpPred
    elif method == 'LH':
        '''
        flow_date = "2019-01-11"
        end_date = "2019-01-01"
        iso = "PJMISO"
        node_id = "51288"
        node_name = "WESTERN HUB"
        product = products.prodInfo("PDA")
        '''
        dalmpPred = dalmp_likehour.dalmp_LH(iso, node_id, "WESTERN HUB", flow_date, flow_date)
        dalmpPred = dalmpPred["onpeak_avg"] if peak_type == "onpeak" else dalmpPred["offpeak_avg"]
        return dalmpPred
    elif method == 'linearensemble':
        '''
        flow_date = "2019-01-15"
        end_date = "2019-01-01"
        iso = "PJMISO"
        node_id = "51288"
        node_name = "WESTERN HUB"
        product = products.prodInfo("PDA")
        peak_type = product.peaktype()
        data_type = "dalmp"
        '''
        LH = dalmp_likehour.dalmp_LH(iso, node_id, "WESTERN HUB", flow_date, flow_date)
        LHdalmpPred = LH["onpeak_avg"] if peak_type == "onpeak" else LH["offpeak_avg"]

        dayzerdaily = dalmp_dayzer.dalmp_pred_vintage(iso=iso,
                                                    node_id=node_id,
                                                    data_type=data_type,
                                                    start_date=flow_date,
                                                    end_date=flow_date)
        dayzerdalmpPred = dayzerdaily["onpeak_pred"] if peak_type == "onpeak" else dayzerdaily["offpeak_pred"]

        if LHdalmpPred.shape[0] ==0 :
            return LHdalmpPred
        elif dayzerdalmpPred.shape[0] == 0:
            return dayzerdalmpPred
        else:
            return (LHdalmpPred*0.4 + dayzerdalmpPred*0.6)

    elif method == 'MLemsemble':
        '''
        flow_date = "2019-01-11"
        end_date = "2019-01-01"
        iso = "PJMISO"
        node_id = "51288"
        node_name = "WESTERN HUB"
        product = products.prodInfo("PDA")
        '''
        LH = dalmp_likehour.dalmp_LH(iso, node_id, "WESTERN HUB", flow_date, flow_date)
        LHdalmpPred = LH["peak"] if peak_type == "onpeak" else LH["offpeak"]

        dayzerdaily = dalmp_dayzer.dalmp_pred_vintage(iso=iso,
                                                    node_id=node_id,
                                                    data_type=data_type,
                                                    start_date=flow_date,
                                                    end_date=flow_date)
        dayzerdalmpPred = dayzerdaily["onpeak_pred"] if peak_type == "onpeak" else dayzerdaily["offpeak_pred"]

        return LHdalmpPred*0.4 + dayzerdalmpPred*0.6

    else:
        print("Prediction method not found!")
        return None


def backTest(product,
             margin=2,
             maxLongVolume=4000,
             maxShortVolume=-800,
             dateStart="2019-03-20",
             dateEnd="2019-03-20",
             method="DayzerDaily"):
    tick_df = parseTickData(product, dateStart,dateEnd)

    startDateTime = min(tick_df.deal_time)
    endDateTime = max(tick_df.deal_time)
    startDate = startDateTime.date()
    endDate = endDateTime.date()

    # Define dealed tick format
    result = pd.DataFrame({"dealDate": [],
                           "flowDate": [],
                           "hour": [],
                           "dealPrice": [],
                           "volume": [],
                           "notionalValue": []})

    for day in pd.date_range(startDate, endDate, freq='D').tolist():
        # day is the deal date
        print("..... backtesting " + datetime.datetime.strftime(day, "%Y-%m-%d") + ".....")
        dayEnd = day + datetime.timedelta(days=1)
        dailyTick = tick_df.query('deal_time >= @day') \
            .query('deal_time < @dayEnd')
        # initialize daily position
        position = pd.DataFrame({"dealDate": [],
                                 "flowDate": [],
                                 "hour": [],
                                 "dealPrice": [],
                                 "volume": [],
                                 "notionalValue": []})
        # dalmpPredPeak, dalmpPredOffpeak = NN.NN_Model(datePivot=datetime.datetime.strftime(day,"%Y-%m-%d"))


        for hour in range(6, 20):
            # hourlyTick = dailyTick.loc[(dailyTick.deal_time.dt.hour >= hour) & (dailyTick.deal_time.dt.hour < hour + 1)]
            hourlyTick = dailyTick.query('deal_time.dt.hour >= @hour') \
                .query('deal_time.dt.hour <  @hour + 1')
            if hourlyTick.shape[0] == 0:
                # no transactions were found
                continue

            flowDate = hourlyTick.iloc[0]["strip_begin"].date().strftime("%Y-%m-%d")

            # Get hourly prediction
            dalmpPred = getDalmpPred(method=method,
                                     timestamp=hourlyTick.iloc[0]["deal_time"].strftime(
                                         "%Y-%m-%d %H:%M:%S"),
                                     product=product,
                                     flow_date=flowDate)

            if dalmpPred.shape[0] == 0:
                # no predictions were found
                continue

            bidPrice = float(dalmpPred) - margin
            offerPrice = float(dalmpPred) + margin

            for index, tick in hourlyTick.iterrows():
                if (str(tick["bid/offer"]) == "Offer") & (float(tick["order_price"]) <= float(bidPrice)):
                    currPosition = position.volume.sum()
                    if currPosition < maxLongVolume:
                        dealVolume = min(maxLongVolume - currPosition, tick["total_volume"])
                        position = position.append({"dealDate": day.date(),
                                                    "flowDate": flowDate,
                                                    "hour": hour,
                                                    "dealPrice": tick["order_price"],
                                                    "volume": dealVolume,
                                                    "notionalValue": -tick["order_price"] * dealVolume,
                                                    "dealTime": tick["deal_time"]},
                                                   ignore_index=True)
                elif (str(tick["bid/offer"]) == "Bid") & (float(tick["order_price"]) >= float(offerPrice)):
                    currPosition = position.volume.sum()
                    if currPosition > maxShortVolume:
                        dealVolume = - min(currPosition - maxShortVolume, tick["total_volume"])
                        position = position.append({"dealDate": day.date(),
                                                    "flowDate": flowDate,
                                                    "hour": hour,
                                                    "dealPrice": tick["order_price"],
                                                    "volume": dealVolume,
                                                    "notionalValue": -tick["order_price"] * dealVolume,
                                                    "dealTime": tick["deal_time"]},
                                                   ignore_index=True)
                    # else:
                    #   position = position.append({"dealPrice": tick["order_price"],
                    #                               "volume": -currPosition,
                    #                              "notionalValue": tick["order_price"] * tick["order_price"]},
                    #                            ignore_index=True)
        result = result.append(position, sort=False)
    return result


'''
Evaluate the performance
'''
def performance(product, tickDataFrame):
    if tickDataFrame.shape[0] == 0:
        return
    peakType = "onpeak_avg" if product.peaktype() == "onpeak" else "offpeak_avg"
    dailyResult = tickDataFrame.assign(type = np.where(tickDataFrame.volume > 0, "Long","Short")) \
                               .pipe(util.select,["flowDate", "type","volume", "notionalValue"])

    startFlowDate = min(dailyResult.flowDate)
    endFlowDate = max(dailyResult.flowDate)

    histDalmp = dalmp_dayzer.dalmp_real(product.iso(),
                                        product.nodeid(),
                                        "dalmp",
                                        startFlowDate,
                                        endFlowDate)[["flow_date", peakType]] \
        .rename(columns={peakType: 'settleDalmp',
                         'flow_date': 'flowDate'})

    histDalmp.flowDate = histDalmp.flowDate.astype(str)

    dayzerDalmp = dalmp_dayzer\
        .dalmp_pred_vintage(product.iso(),product.nodeid(),"dalmp",startFlowDate,endFlowDate)\
        .pipe(util.select,['flow_date','onpeak_pred' if product.peaktype() == "onpeak" else "offpeak_pred" ])\
        .rename(columns={'flow_date':'flowDate'})

    dayzerDalmp.flowDate =   dayzerDalmp.flowDate.astype(str)

    likehourDalmp = dalmp_likehour\
        .dalmp_LH(product.iso(),product.nodeid(),"WESTERN HUB",startFlowDate,endFlowDate)\
        .pipe(util.select,['flow_date',peakType])\
        .rename(columns = {peakType:'likehour_pred',
                           'flow_date':'flowDate'})

    likehourDalmp.flowDate = likehourDalmp.flowDate.astype(str)

    dailyResult = dailyResult\
        .merge(histDalmp, how='left', on='flowDate')\
        .merge(likehourDalmp, how='left', on='flowDate')\
        .merge(dayzerDalmp,how='left', on='flowDate')\
        .assign(pred_delta = lambda df: df.likehour_pred - df.onpeak_pred)

    winRate = dailyResult.assign(type=np.where(dailyResult.volume > 0, "Long", "Short"),
                       PnL = lambda df: df.notionalValue + df.volume * df.settleDalmp,
                       win = np.where(dailyResult.notionalValue + dailyResult.volume * dailyResult.settleDalmp > 0, 1, 0)) \
        .pipe(util.select,["type","win"])\
        .groupby('type')\
        .agg(['sum','count'])\
        .reset_index()
    winRate.columns = ['type','win','total']
    winRate = winRate.assign(winrate= lambda df: df.win/df.total)
    winrateshort = 0 if winRate.query("type=='Short'").winrate.shape[0] ==0 else float(winRate.query("type=='Short'").winrate)
    winratelong = 0 if winRate.query("type=='Long'").winrate.shape[0] ==0 else float(winRate.query("type=='Long'").winrate)
    winRate = {'long':winratelong,'short':winrateshort}
    dailyResult = dailyResult.assign(type = np.where(dailyResult.volume > 0, "Long", "Short"),
                       PnL = lambda df: df.notionalValue + df.volume * df.settleDalmp) \
        .groupby(['flowDate','type'])\
        .agg({'volume':['sum','count'],
              'PnL':['sum']}) \
        .reset_index()


    dailyResult.columns = ["flowDate","type","volume","count","PnL"]

    if set(dailyResult.type) == {'Short'}:
        dailyResult = dailyResult.rename(columns={"volume":"volume_short",
                                                  'count':'count_short',
                                                  'PnL':'PnL_short'})\
            .assign(volume_long = 0,
                    count_long = 0,
                    PnL_long = 0)\
            .pipe(util.select,["flowDate",
                           "volume_long",
                           "volume_short",
                           "count_long",
                           "count_short",
                           "PnL_long",
                           "PnL_short"])
    elif set(dailyResult.type) == {'Long'}:
        dailyResult = dailyResult.rename(columns={"volume": "volume_long",
                                                  'count': 'count_long',
                                                  'PnL': 'PnL_long'}) \
            .assign(volume_short=0,
                    count_short=0,
                    PnL_short=0) \
            .pipe(util.select, ["flowDate",
                                "volume_long",
                                "volume_short",
                                "count_long",
                                "count_short",
                                "PnL_long",
                                "PnL_short"])
    else:
        dailyResult = dailyResult.pivot(index = 'flowDate',columns = 'type').reset_index()

        dailyResult.columns = ["flowDate",
                               "volume_long",
                               "volume_short",
                               "count_long",
                               "count_short",
                               "PnL_long",
                               "PnL_short"]

    return dailyResult,winRate
