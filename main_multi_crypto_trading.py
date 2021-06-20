"""Training and Backtesting Crypto Trading Bot with
   Binance Historycal Data
"""

import os

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

import itertools

from finrl.config import config_crypt as config
from finrl.marketdata.binance_data import BinanceData
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline

if __name__ == "__main__":

    # Create Folders
    if not os.path.exists("./" + config.DATA_DIR):
        os.makedirs("./" + config.DATA_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

    # Download Data
    print("start date:",config.START_DATE)
    print("end date:",config.END_DATE)
    print("data_dir:",config.DATA_DIR)
    print("list of tickers:",config.BINANCE_TICKER)

    df = BinanceData(start_date = config.START_DATE,
                     end_date = config.END_DATE,
                     data_dir = config.DATA_DIR,
                     ticker_list = config.BINANCE_TICKER).fetch_data()
    
    print("Shape of the downloaded dataframe:",df.shape)

    print(df.sort_values(['date','tic'],ignore_index=True).head())

    # Preprocess Data
    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                         use_turbulence=True,
                         user_defined_feature = False)
    processed = fe.preprocess_data(df)
    
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])
    processed_full = processed_full.fillna(0)

    print(processed_full.sort_values(['date','tic'],ignore_index=True).head(10))

    import pdb;pdb.set_trace()

    # Design Environment
    train = data_split(processed_full, '2009-01-01','2019-01-01')
    trade = data_split(processed_full, '2019-01-01','2021-01-01')
    print(len(train))
    print(len(trade))

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
            "hmax": 100,
            "initial_amount": 1000000,
            "buy_cost_pct": 0.001,
            "sell_cost_pct": 0.001,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4
    }

    e_train_gym = StockTradingEnv(df = train, **env_kwargs)

    # environment for training
    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))

    # DRL Algos

    # Model 1: A2C
    agent = DRLAgent(env = env_train)
    model_a2c = agent.get_model("a2c")
    trained_a2c = agent.train_model(model=model_a2c,
                                    tb_log_name='a2c',
                                    total_timesteps=100000)

    # Model 2: DDPG
    agent = DRLAgent(env = env_train)
    model_ddpg = agent.get_model("ddpg")
    trained_ddpg = agent.train_model(model=model_ddpg,
                                     tb_log_name='ddpg',
                                     total_timesteps=50000)

    # Model 3: PPO
    agent = DRLAgent(env = env_train)
    PPO_PARAMS = {
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 128,
        
    }
    model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
    trained_ppo = agent.train_model(model=model_ppo,
                                    tb_log_name='ppo',
                                    total_timesteps=50000)

    # Model 4: TD3
    agent = DRLAgent(env = env_train)
    TD3_PARAMS = {"batch_size": 100,
                  "buffer_size": 1000000,
                  "learning_rate": 0.001}
    model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)
    trained_td3 = agent.train_model(model=model_td3,
                                    tb_log_name='td3',
                                    total_timesteps=30000)

    # Model 5: SAC
    agent = DRLAgent(env = env_train)
    SAC_PARAMS = {
            "batch_size": 128,
            "buffer_size": 1000000,
            "learning_rate": 0.0001,

            "learning_starts": 100,
            "ent_coef": "auto_0.1",
        
    }
    model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)
    trained_sac = agent.train_model(model=model_sac,
                                    tb_log_name='sac',
                                    total_timesteps=80000)

    # Trading

    # Set turbulence threshold
    data_turbulence = processed_full[(processed_full.date<'2019-01-01') & (processed_full.date>='2009-01-01')]
    insample_turbulence = data_turbulence.drop_duplicates(subset=['date'])

    turbulence_threshold = np.quantile(insample_turbulence.turbulence.values,1)
    print("Turbulence Threshold:",turbulence_threshold)

    # Trade
    trade = data_split(processed_full, '2019-01-01','2021-01-01')
    e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 380, **env_kwargs)
    # env_trade, obs_trade = e_trade_gym.get_sb_env()

    df_account_value, df_actions = DRLAgent.DRL_prediction(
            model=trained_sac,
            environment = e_trade_gym
    )

    # save account value and df_actions

    # Backtesting
    print("==============Get Backtest Results===========")
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

    perf_stats_all = backtest_stats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')

    #baseline stats
    print("==============Get Baseline Stats===========")
    baseline_df = get_baseline(
                ticker="^DJI",
                start = '2019-01-01',
                end = '2021-01-01'
    )
    stats = backtest_stats(baseline_df, value_col_name = 'close')

    print("==============Compare to DJIA===========")
    #%matplotlib inline
    # S&P 500: ^GSPC
    # Dow Jones Index: ^DJI
    # NASDAQ 100: ^NDX
    backtest_plot(df_account_value,
                  baseline_ticker = '^DJI',
                  baseline_start = '2019-01-01',
                  baseline_end = '2021-01-01')

    
