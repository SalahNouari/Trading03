
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import datetime

# The above could be sent to an independent module
import backtrader as bt
from backtrader import (SQN, AnnualReturn, TimeReturn, SharpeRatio, TradeAnalyzer)
from TestModules.Strategies import ST_test_001  as st0
import LoadArchive as LArc
from ConfigDbSing import ConfigDbSing
import pandas as pd

def loadData():
		_connect_db = ConfigDbSing().get_config()
		pref_comp = _connect_db["comp_pref"]

		_data = LArc.LoadArchive.Picle(f"{pref_comp}Trading03\\Data\\Sber\\candles1H.pickle")
		_pd = _data.Pd

		# data=yf.download('SPY', start='2021-01-01', end='2021-12-31')
		_pd['datetime'] = pd.to_datetime(_pd['date'].astype(str) + ' ' + _pd['time'].astype(str))
		_pd.drop(['date', 'time'], axis=1, inplace=True)
		# _pd['datetime'] = pd.Series(_pd['datetime'])
		_pd = _pd.set_index('datetime')
		return  _pd

def runstrategy(_pd):
  args = parse_args()
  args.plot = True
  # Create a cerebro
  cerebro = bt.Cerebro()

  # Get the dates from the args
  fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
  todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')

  # df = Db_form_data("Trading02", formatd=2) # timeframe='1min'  formatd=2 - для backtrader
  df = _pd
  _count_df = len(df)
  x = df.index.tolist()[0]

  data = bt.feeds.PandasData(dataname=df, nocase=True,)

  # Add the 1st data to cerebro
  cerebro.adddata(data)


  cerebro.addstrategy(st0.LongShortStrategy,    # Add the strategy
                      period=args.period,
                      onlylong=args.onlylong,
                      csvcross=args.csvcross,
                      stake=args.stake)

  # Add the commission - only stocks like a for each operation
  cerebro.broker.setcash(args.cash)

  # Add the commission - only stocks like a for each operation
  cerebro.broker.setcommission(commission=args.comm,
                               mult=args.mult,
                               margin=args.margin)

  tframes = dict(
    days=bt.TimeFrame.Days,
    weeks=bt.TimeFrame.Weeks,
    months=bt.TimeFrame.Months,
    years=bt.TimeFrame.Years)

  # Add the Analyzers
  cerebro.addanalyzer(SQN)
  if args.legacyannual:
    cerebro.addanalyzer(AnnualReturn)
    cerebro.addanalyzer(SharpeRatio, legacyannual=True)
  else:
    cerebro.addanalyzer(TimeReturn, timeframe=tframes[args.tframe])
    cerebro.addanalyzer(SharpeRatio, timeframe=tframes[args.tframe])

  cerebro.addanalyzer(TradeAnalyzer)

  cerebro.addwriter(bt.WriterFile, csv=args.writercsv, rounding=4)

  # And run it
  cerebro.run()

  # Plot if requested
  cerebro.plot(numfigs=args.numfigs, volume=False, zdown=False)

  # if args.plot:
  #   cerebro.plot(numfigs=args.numfigs, volume=False, zdown=False)

def parse_args():
    parser = argparse.ArgumentParser(description='TimeReturn')
    path = 'E:\MLserver\Trading01\BackTrader\datas/2005-2006-day-001.txt'
    parser.add_argument('--data', '-d',
                        default=path,
                        # default='../../datas/2005-2006-day-001.txt',
                        help='data to add to the system')

    parser.add_argument('--fromdate', '-f',
                        default='2005-01-01',
                        help='Starting date in YYYY-MM-DD format')

    parser.add_argument('--todate', '-t',
                        default='2006-12-31',
                        help='Starting date in YYYY-MM-DD format')

    parser.add_argument('--period', default=15, type=int,
                        help='Period to apply to the Simple Moving Average')

    parser.add_argument('--onlylong', '-ol', action='store_true',
                        help='Do only long operations')

    parser.add_argument('--writercsv', '-wcsv', action='store_true',
                        help='Tell the writer to produce a csv stream')

    parser.add_argument('--csvcross', action='store_true',
                        help='Output the CrossOver signals to CSV')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--tframe', default='years', required=False,
                       choices=['days', 'weeks', 'months', 'years'],
                       help='TimeFrame for the returns/Sharpe calculations')

    group.add_argument('--legacyannual', action='store_true',
                       help='Use legacy annual return analyzer')

    parser.add_argument('--cash', default=100000, type=int,
                        help='Starting Cash')

    parser.add_argument('--comm', default=2, type=float,
                        help='Commission for operation')

    parser.add_argument('--mult', default=10, type=int,
                        help='Multiplier for futures')

    parser.add_argument('--margin', default=2000.0, type=float,
                        help='Margin for each future')

    parser.add_argument('--stake', default=1, type=int,
                        help='Stake to apply in each operation')

    parser.add_argument('--plot', '-p', action='store_true',
                        help='Plot the read data')

    parser.add_argument('--numfigs', '-n', default=1,
                        help='Plot using numfigs figures')

    return parser.parse_args()


if __name__ == '__main__':
  print("---====> Test ST-001 <===---")

# _st0 = st0.LongShortStrategy()
  runstrategy(loadData())
