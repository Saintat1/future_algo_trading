import importlib
import pandas as pd
import API.dalmp_Dayzer as dalmp

importlib.reload(dalmp)

realDalmp = dalmp.dalmp_real("PJMISO",
                             "51288",
                              "dalmp",
                             "2019-03-20",
                             "2019-04-23",
                             hourly=True)

realDalmp = pd.melt(realDalmp,
                    id_vars='flow_date',
                    value_vars=["he" + str(i) for i in range(1, 25)],
                    value_name="DALMP")

realDalmp = realDalmp.rename(columns={'variable': "hour", }) \
    .assign(hour=lambda df: df.hour.str[2:].astype(int)) \
    .sort_values(by=["flow_date", "hour"])

predictDalmp = dalmp.dalmp_pred("PJMISO", "51288", "dalmp", "2019-03-20", "2019-04-23", hourly=True)

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

xxx = dalmpPred.assign(APE=lambda df: abs(df.dalmp - df.predictDalmp) / df.dalmp,
                       SE=lambda df: (df.dalmp - df.predictDalmp) ** 2,
                       AE=lambda df: abs(df.dalmp - df.predictDalmp))

xxx.APE.mean()

MAPE = finalDF.APE.mean()
MSE = finalDF.SE.mean()
MAE = finalDF.AE.max()
