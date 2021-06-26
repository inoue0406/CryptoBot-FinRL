import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        default='sac',
        help='Name of RL models')
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Data directory path')
    parser.add_argument(
        '--results_dir',
        type=str,
        help='Results directory path')
    parser.add_argument(
        '--start_date',
        type=str,
        help='Start date of the market dataset')
    parser.add_argument(
        '--end_date',
        type=str,
        help='End date of the market dataset')
    parser.add_argument(
        '--start_trade_date',
        type=str,
        help='Trading (Testing) date of the market dataset')
    parser.add_argument(
        '--tech_indicators',
        type=str,
        nargs='*',
        help='Technical indicators to be used as the input of RL model')
    parser.add_argument(
        '--market',
        type=str,
        help='Which market data to be used')
    parser.add_argument(
        '--tickers',
        type=str,
        nargs='*',
        help='Tickers to be used for trading')
    parser.add_argument(
        '--hmax',
        type=float,
        default=100,
        help='Maximum trading volume per trade in USDT')
    parser.add_argument(
        '--initial_amount',
        type=float,
        default=10000,
        help='Initial amount in USDT')
    parser.add_argument(
        '--buy_cost_pct',
        type=float,
        default=0.1,
        help='Cost of buying in %')
    parser.add_argument(
        '--sell_cost_pct',
        type=float,
        default=0.1,
        help='Cost of selling in %')
    
    args = parser.parse_args()

    return args
