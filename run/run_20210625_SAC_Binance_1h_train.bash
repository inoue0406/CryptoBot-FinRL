#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# cd to parent dir
cd ..

case="20210625_SAC_Binance_1h_train"

# parameters

# buy_cost_pct (float): cost for buying shares
# sell_cost_pct (float): cost for selling shares
# hmax (int, array): maximum cash to be traded in each tradeper asset.
#                   If an array is provided, then each index correspond to each asset

# running script
python main_multi_crypto_trading.py --no_train --prep_flag 0\
       --model_name sac\
       --data_dir daata/Binance/spot-1min\
       --results_dir results/$case\
       --start_date 2020-02-21\
       --end_date 2021-06-11\
       --start_trade_date 2021-05-01\
       --tech_indicators macd boll_ub boll_lb rsi_30 cci_30 dx_30 close_30_sma close_60_sma\
       --market Binance\
       --tickers BTCUSDT ETHUSDT ADAUSDT XRPUSDT DOGEUSDT LTCUSDT LINKUSDT\
       --hmax 100\
       --initial_amount 10000\
       --buy_cost_pct 0.001\
       --sell_cost_pct 0.001\
       --learning_rate 0.000001\
       --total_timesteps 800000\
