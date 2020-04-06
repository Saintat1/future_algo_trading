import datetime
import json
from functools import reduce

import holidays
import pandas as pd


def datareformat(jsonName="test.json", datePivot="2019-03-01"):
    '''
    :param : read Like hour rds file
    :return: split
    '''

    def series_to_supervised(data, n_in=1, n_out=1, dropnan=False):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('%s(t-%d)' % (list(data)[j], i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('%s(t)' % (list(data)[j])) for j in range(n_vars)]
            else:
                names += [('%s(t+%d)' % (list(data)[j], i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    with open("./Algo_bidprice/backtest/" + jsonName) as json_file:
        data = json.load(json_file)

    dayzerData = data["data"]

    dalmp = pd.DataFrame(dayzerData["yes_ts"]["DALMP"]["WESTERN HUB"])

    astrolight = None
    for i in dayzerData:  # i= astro_light, ms_ts ...
        subsetData = dayzerData[i]
        if i == "astro_light":
            astroLight = subsetData["sunlight"]
            for location in astroLight:
                objectDF = pd.DataFrame(astroLight[location])
                if astrolight is None:
                    astrolight = objectDF
                else:
                    astrolight = pd.merge(astrolight, objectDF, how='left', on="date.he")

    tempfc = dayzerData["yes_ts"]["WSIFC_TMP_BIDCLOSE"]
    t = None

    for i in tempfc:
        if t is None:
            t = pd.DataFrame(tempfc[i])
        else:
            t = pd.merge(t, pd.DataFrame(tempfc[i]), how='left', on="date.he")

    tempfc = t

    ngPrice = dayzerData["ms_ts_daily"]["Close"]

    t = None

    for i in ngPrice:
        if t is None:
            t = pd.DataFrame(ngPrice[i])
        else:
            t = pd.merge(t, pd.DataFrame(ngPrice[i]), how='left', on="date.he")

    ngPrice = t

    avgCloudCover = dayzerData["ms_ts"]["AverageCloudCover"]

    t = None

    for i in avgCloudCover:
        if t is None:
            t = pd.DataFrame(avgCloudCover[i])
        else:
            t = pd.merge(t, pd.DataFrame(avgCloudCover[i]), how='left', on="date.he")

    avgCloudCover = t

    Precipitation = dayzerData["ms_ts"]["Precipitation"]

    t = None

    for i in Precipitation:
        if t is None:
            t = pd.DataFrame(Precipitation[i])
        else:
            t = pd.merge(t, pd.DataFrame(Precipitation[i]), how='left', on="date.he")

    Precipitation = t

    AvgRelativeHumidity = dayzerData["ms_ts"]["AvgRelativeHumidity"]

    t = None

    for i in AvgRelativeHumidity:
        if t is None:
            t = pd.DataFrame(AvgRelativeHumidity[i])
        else:
            t = pd.merge(t, pd.DataFrame(AvgRelativeHumidity[i]), how='left', on="date.he")

    AvgRelativeHumidity = t

    loadForecast = dayzerData["yes_ts"]["LOAD_FORECAST"]

    t = None

    for i in loadForecast:
        if t is None:
            t = pd.DataFrame(loadForecast[i])
        else:
            t = pd.merge(t, pd.DataFrame(loadForecast[i]), how='left', on="date.he")

    loadForecast = t

    finalDF = [ngPrice, avgCloudCover, Precipitation, AvgRelativeHumidity, tempfc, astrolight, loadForecast, dalmp]

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['date.he'], how='outer'), finalDF)

    df_merged_copy = df_merged.copy()
    date_he = df_merged_copy.pop("date.he")
    dalmp_column = df_merged_copy.pop("yes_ts.DALMP.WESTERN HUB")

    df_merged_copy = series_to_supervised(df_merged_copy, n_in=1 , n_out=1, dropnan=False)

    df_merged_copy = pd.DataFrame(pd.concat([date_he, df_merged_copy, dalmp_column], axis=1))

    df_merged_copy.dropna(inplace=True)

    us_holidays = holidays.US()


    # exclude weekends and holiday

    # Used when do backtest, otherwise comment out
    df_merged_copy["date"] = df_merged_copy["date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

    df_merged_copy = df_merged_copy[df_merged_copy.holiday != 1]
    df_merged_copy = df_merged_copy.loc[df_merged_copy['holiday'].isin([0, 1, 2, 3, 4])]
    df_merged_copy.drop(labels=["holiday"], inplace=True, axis=1)

    hour = df_merged_copy.pop('hour')
    hourList = [i for i in range(1, 25)]
    for hours in hourList:
        df_merged_copy["he" + str(hours)] = (hour == hours) * 1.0

    weekday = df_merged_copy.pop('weekday')
    weekdayList = [i for i in range(5)]
    for weekdays in weekdayList:
        df_merged_copy["weekday" + str(weekdays)] = (weekday == weekdays) * 1.0

    month = df_merged_copy.pop('month')
    monthList = [i for i in range(1, 13)]
    for months in monthList:
        df_merged_copy["month" + str(months)] = (month == months) * 1.0

    df_merged_copy.dropna(inplace=True)
    # df_merged_copy.drop(labels=["date.he"], inplace=True, axis=1)
    df_merged_copy.set_index('date.he', inplace=True)

    df_merged_copy_train = df_merged_copy[
        (df_merged_copy["date"] < datetime.datetime.strptime(datePivot, "%Y-%m-%d"))]

    df_merged_copy_test = df_merged_copy[
        (df_merged_copy["date"] >= datetime.datetime.strptime(datePivot, "%Y-%m-%d"))]

    df_merged_copy_train.pop("date")
    df_merged_copy_test.pop("date")

    return df_merged_copy_train, df_merged_copy_test
