"""Contains methods and classes to collect data from
Binance Historical Data
"""

import pandas as pd
import glob
import urllib
import requests

import time
import datetime

class BinanceData_dl:
    """Binance Data Class

    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        ticker_list : list
            a list of stock tickers (modified from config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, data_dir: str, ticker_list: list):

        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        self.ticker_list = ticker_list

    def binance_klines_fromurl(self, symbol, interval, start, end, limit=5000):
        # convert to unix time in [ms]
        start_unix = int(time.mktime(datetime.datetime.strptime(start, "%Y-%m-%d").timetuple())*1000)
        end_unix = int(time.mktime(datetime.datetime.strptime(end, "%Y-%m-%d").timetuple())*1000)
        df = pd.DataFrame()
        startDate = end_unix
        while startDate>start_unix:
            url = 'https://api.binance.com/api/v3/klines?symbol=' + \
                  symbol + '&interval=' + interval + '&limit='  + str(limit)
            if startDate is not None:
                url += '&endTime=' + str(startDate)
                
            print("accessing URL:",url)
            data = requests.get(url).json()
            df2 = pd.DataFrame(data)
            df2.columns = ['unix', 'open', 'high', 'low', 'close', 'volume', 'Closetime', 'Quote asset volume','Number of trades','Taker by base', 'Taker buy quote', 'Ignore']
            df2 = df2.iloc[:-1] # avoid duplicates
            #
            df2["date"] = pd.to_datetime(df2["unix"],unit="ms")
            df2["tic"] = symbol
            df = pd.concat([df2, df], axis=0, ignore_index=True, keys=None)
            startDate = df.unix[0]
        df.reset_index(drop=True, inplace=True)
        # check duplicates
        print("check duplicates:",sum(df["date"].duplicated()))
        # convert to numeric
        df["open"] = pd.to_numeric(df["open"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["close"] = pd.to_numeric(df["close"])
        df["volume"] = pd.to_numeric(df["volume"])
            
        return df

    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from Binance Pre-downloaded Data

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        
        interval = "1h"
            
        # column names to be used
        col_use = ["date","tic","open","high","low","close","volume"]

        # create dataframe with regular interval
        if interval == "1m":
            df_regular = pd.DataFrame({"date":pd.date_range(start=self.start_date,
                                                            end=self.end_date,
                                                            freq="Min")})
        elif interval == "1h":
            df_regular = pd.DataFrame({"date":pd.date_range(start=self.start_date,
                                                           end=self.end_date,
                                                            freq="H")})
        else:
            print("unsupported interval",interval)
        df_regular = df_regular[:-1]
        
        id_nomiss = pd.Series([True]*len(df_regular))

        df_all = pd.DataFrame()
        
        for tic in self.ticker_list:
            
            df = self.binance_klines_fromurl(tic, interval, self.start_date, self.end_date)
            df = df[col_use]
            # create day of the week column (monday = 0)
            #df["date"] = pd.to_datetime(df["date"])
            df["day"] = df["date"].dt.dayofweek

            label = df["tic"][0]
            start = sorted(df["date"])[0]
            end = sorted(df["date"])[-1]
            
            if not(label in self.ticker_list):
                print ("skipped:",label)
                continue
            
            # extract from start_date to end_date
            id_slct = (df["date"] >= self.start_date) & \
                      (df["date"] <  self.end_date)
            print("No. of selected data",sum(id_slct))
            df = df[id_slct]

            df_m = pd.merge(df_regular,df,how="left")

            # detect missing time series
            id_nomiss = id_nomiss & (~df_m["open"].isnull()) # ~ is "NOT"
            print("No. of missing entries",sum(df_m["open"].isnull()))
                        
            print("ticker:%s data period %s - %s" % (label,start,end))
            #print("ticker:%s data period %s - %s" % (label,df["date"].values[0],df["date"].values[-1]))
            df_all = df_all.append(df_m)
            
        # remove missing data
        df_all = df_all.loc[id_nomiss]

        # convert date to standard string format, easy to filter
        df_all["date"] = df_all["date"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        
        print("Shape of DataFrame: ", df_all.shape)
        print("Display DataFrame: ", df_all.head())

        df_all = df_all.sort_values(by=['date','tic']).reset_index(drop=True)

        return df_all

class BinanceData:
    """Binance Data Class

    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        ticker_list : list
            a list of stock tickers (modified from config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, data_dir: str, ticker_list: list):

        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from Binance Pre-downloaded Data

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Read the data in a pandas DataFrame:

        files = glob.glob(self.data_dir + "/*csv")

        # column names to be used
        col_use = ["date","symbol","open","high","low","close","volume"]

        # create dataframe with regular interval
        df_regular = pd.DataFrame({"date":pd.date_range(start=self.start_date,
                                                        end=self.end_date,
                                                        freq="Min")})
        df_regular = df_regular[:-1]
        
        id_nomiss = pd.Series([True]*len(df_regular))

        df_all = pd.DataFrame()
        
        for file in files:
            print("opening csv file:",file)
            df = pd.read_csv(file,skiprows=1)
            df = df[col_use]
            # rename to be consistent with FinRL 
            df = df.rename(columns={'symbol': 'tic', 'Volume USDT': 'volume'})
            # create day of the week column (monday = 0)
            df["date"] = pd.to_datetime(df["date"])
            df["day"] = df["date"].dt.dayofweek

            label = df["tic"][0]
            start = sorted(df["date"])[0]
            end = sorted(df["date"])[-1]
            
            if not(label in self.ticker_list):
                print ("skipped:",label)
                continue
            
            # extract from start_date to end_date
            id_slct = (df["date"] >= self.start_date) & \
                      (df["date"] <  self.end_date)
            print("No. of selected data",sum(id_slct))
            df = df[id_slct]

            df_m = pd.merge(df_regular,df,how="left")

            # detect missing time series
            id_nomiss = id_nomiss & (~df_m["open"].isnull()) # ~ is "NOT"
            print("No. of missing entries",sum(df_m["open"].isnull()))
                        
            print("ticker:%s data period %s - %s" % (label,start,end))
            #print("ticker:%s data period %s - %s" % (label,df["date"].values[0],df["date"].values[-1]))
            df_all = df_all.append(df_m)
            
        # remove missing data
        df_all = df_all.loc[id_nomiss]

        # convert date to standard string format, easy to filter
        df_all["date"] = df_all["date"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        
        print("Shape of DataFrame: ", df_all.shape)
        print("Display DataFrame: ", df_all.head())

        df_all = df_all.sort_values(by=['date','tic']).reset_index(drop=True)

        return df_all

