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
from opts import parse_opts

from finrl.marketdata.binance_data import BinanceData_dl
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline

from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3 import PPO

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

def run_prep(opt,fname_processed):
    # Run prep routines
    opt = parse_opts()
    
    # Download Data
    print("start date:",opt.start_date)
    print("end date:",opt.end_date)
    print("data_dir:",opt.start_trade_date)
    print("list of tickers:",opt.tickers)

    if opt.market == "Binance":
        df = BinanceData_dl(start_date = opt.start_date,
                            end_date = opt.end_date,
                            data_dir = opt.data_dir,
                            ticker_list = opt.tickers).fetch_data()
    
    print("Shape of the downloaded dataframe:",df.shape)

    print(df.sort_values(['date','tic'],ignore_index=True).head())

    # Preprocess Data
    print("Feature Engineering:")
    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list = opt.tech_indicators,
                         #use_turbulence=True,
                         use_turbulence=False,
                         user_defined_feature = False)
    processed = fe.preprocess_data(df)

    # save processed data
    processed.to_csv(fname_processed)

    processed_full = processed
    processed_full = processed_full.sort_values(['date','tic']).reset_index(drop=True)
    
    print(processed_full.sort_values(['date','tic'],ignore_index=True).head(10))
    
    return processed_full


def drl_train(env_train, model_type, opt):
    # do the training of DRL models
    
    if model_type == "a2c":
        # Model 1: A2C
        agent = DRLAgent(env_train, opt.results_dir)
        model_a2c = agent.get_model("a2c")
        trained_model = agent.train_model(model=model_a2c,
                                        tb_log_name='a2c',
                                        total_timesteps=100000)
    elif model_type == "ddpg":
        # Model 2: DDPG
        agent = DRLAgent(env_train, opt.results_dir)
        model_ddpg = agent.get_model("ddpg")
        trained_model = agent.train_model(model=model_ddpg,
                                         tb_log_name='ddpg',
                                         total_timesteps=50000)
    elif model_type == "ppo":
        # Model 3: PPO
        agent = DRLAgent(env_train, opt.results_dir)
        PPO_PARAMS = {
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 128,
            
        }
        model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
        trained_model = agent.train_model(model=model_ppo,
                                        tb_log_name='ppo',
                                        total_timesteps=50000)
    elif model_type == "td3":
        # Model 4: TD3
        agent = DRLAgent(env_train, opt.results_dir)
        TD3_PARAMS = {"batch_size": 100,
                      "buffer_size": 1000000,
                      "learning_rate": 0.001}
        model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)
        trained_model = agent.train_model(model=model_td3,
                                        tb_log_name='td3',
                                        total_timesteps=30000)
    elif model_type == "sac":
        # Model 5: SAC
        agent = DRLAgent(env_train, opt.results_dir)
        SAC_PARAMS = {
            "batch_size": 128,
            "buffer_size": 1000000,
            "learning_rate": opt.learning_rate,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
            
        }
        model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)
        trained_model = agent.train_model(model=model_sac,
                                          tb_log_name='sac',
                                          total_timesteps=opt.total_timesteps)
    return trained_model

if __name__ == "__main__":
    # parse command-line options
    opt = parse_opts()
    print(opt)

    # Create Folders
    if not os.path.exists("./" + opt.results_dir):
        os.makedirs("./" + opt.results_dir)

    # Data Prep
    #fname_processed ="data/Binance/preprocessed_binance_1min_single.csv"
    fname_processed ="data/Binance/preprocessed_binance_1hour.csv"
    
    if opt.prep_flag == 1:
        # run the prep
        processed_full = run_prep(opt,fname_processed)
    else:
        # load preprocessed data
        print("loading from preprocessed data")
        processed_full = pd.read_csv(fname_processed)
        
        
    # Design Environment
    train = data_split(processed_full, opt.start_date, opt.start_trade_date)
    trade = data_split(processed_full, opt.start_trade_date, opt.end_date)
    print(len(train))
    print(len(trade))
    
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(opt.tech_indicators)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    
    env_kwargs = {
        "hmax": opt.hmax,
        "initial_amount": opt.initial_amount,
        "buy_cost_pct": opt.buy_cost_pct,
        "sell_cost_pct": opt.sell_cost_pct,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": opt.tech_indicators,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    
    if not(opt.no_train):
        # Training

        e_train_gym = StockTradingEnv(df = train, **env_kwargs)

        # environment for training
        env_train, _ = e_train_gym.get_sb_env()
        print(type(env_train))

        # Training
        trained_model = drl_train(env_train, opt.model_name, opt)
        # save the trained model
        trained_model.save(opt.results_dir + "/trained_model")
        
    else:
        # load pretrained model
        print("loading from pretrained model")
        trained_model = MODELS[opt.model_name].load(opt.results_dir + "/trained_model")

    # Trading
    
    trade = data_split(processed_full, opt.start_trade_date, opt.end_date)

    # with no turbulence threshold
    e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)
    # env_trade, obs_trade = e_trade_gym.get_sb_env()

    df_account_value, df_actions = DRLAgent.DRL_prediction(
            model=trained_model,
            environment = e_trade_gym
    )

    # save account value and df_actions

    # Backtesting
    print("==============Get Backtest Results===========")

    perf_stats_all = backtest_stats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + opt.results_dir + "/perf_stats_all.csv")

    #baseline stats
    print("==============Get Baseline Stats===========")
    baseline_df = get_baseline(
                ticker="BTCUSDT",
                start = opt.start_trade_date,
                end = opt.end_date
    )
    stats = backtest_stats(baseline_df, value_col_name = 'close')

    print("==============Compare to DJIA===========")
    #%matplotlib inline
    backtest_plot(df_account_value,
                  baseline_ticker = "BTCUSDT",
                  baseline_start = opt.start_trade_date,
                  baseline_end = opt.end_date)

    
