import pathlib

# import finrl

import pandas as pd
import datetime
import os

TRAINED_MODEL_DIR = f"trained_models"
# DATASET_DIR = PACKAGE_ROOT / "data"

DATA_DIR = f"data/Binance/spot-1min"
#DATA_DIR = f"data/Binance/spot-1hour"
TRAINED_MODEL_DIR = f"trained_models"
TENSORBOARD_LOG_DIR = f"tensorboard_log"
RESULTS_DIR = f"results"

## time_fmt = '%Y-%m-%d'
START_DATE = "2020-02-21"
END_DATE = "2021-06-11"

START_TRADE_DATE = "2021-05-01"

## dataset default columns
DEFAULT_DATA_COLUMNS = ["date", "tic", "close"]

## stockstats technical indicator column names
## check https://pypi.org/project/stockstats/ for different names
TECHNICAL_INDICATORS_LIST = ["macd","boll_ub","boll_lb","rsi_30", "cci_30", "dx_30","close_30_sma","close_60_sma"]

############## Stock Ticker Setup starts ##############

# Binance TICKER
BINANCE_TICKER = [
    "BTCUSDT",
    "ETHUSDT",
    "ADAUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "LTCUSDT",
    "LINKUSDT"]

#BINANCE_TICKER = [
#    "ADA/USDT",
#    "BNB/USDT",
#    "BTC/USDT",
##    "BTT/USDT",
#    "DASH/USDT",
#    "EOS/USDT",
#    "ETC/USDT",
#    "ETH/USDT",
#    "LINK/USDT",
#    "LTC/USDT",
#    "NEO/USDT",
#    "QTUM/USDT",
#    "TRX/USDT",
#    "XLM/USDT",
#    "XRP/USDT",
#    "ZEC/USDT"]


#BINANCE_TICKER = [
#    "BTC/USDT"]
