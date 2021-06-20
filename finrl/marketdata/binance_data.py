"""Contains methods and classes to collect data from
Binance Historical Data
"""

import pandas as pd
import glob

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
        col_use = ["date","symbol","open","high","low","close","Volume USDT"]

        df_all = pd.DataFrame()
        for file in files:
            print("opening csv file:",file)
            df = pd.read_csv(file,skiprows=1)
            df = df[col_use]
            # rename to be consistent with FinRL 
            df = df.rename(columns={'symbol': 'tic', 'Volume USDT': 'volume'})
            # create day of the week column (monday = 0)
            df["day"] = (pd.to_datetime(df["date"])).dt.dayofweek

            label = df["tic"][0]
            start = sorted(df["date"])[0]
            end = sorted(df["date"])[-1]
            
            if not(label in self.ticker_list):
                print ("skipped:",label)
                continue

            # extract from start_date to end_date
            id_slct = (pd.to_datetime(df["date"]) >= pd.to_datetime(self.start_date)) * \
                      (pd.to_datetime(df["date"]) <  pd.to_datetime(self.end_date))
            print("No. of selected data",sum(id_slct))
            df = df[id_slct]

            # detect missing time series
            delt = pd.to_datetime(df["date"]) - pd.to_datetime(df["date"].shift(1))
            flag = delt.dt.total_seconds() < -60.1
            print(df.loc[flag])
            
            print("ticker:%s data period %s - %s" % (label,start,end))
            print("ticker:%s data period %s - %s" % (label,df["date"].values[0],df["date"].values[-1]))
            df_all = df_all.append(df)
            
        import pdb;pdb.set_trace()
        print("Shape of DataFrame: ", df_all.shape)
        print("Display DataFrame: ", df_all.head())

        df_all = df_all.sort_values(by=['date','tic']).reset_index(drop=True)

        return df_all

