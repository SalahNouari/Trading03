# https://github.com/matplotlib/mplfinance
import talib
import mplfinance as mpf
# from talib import abstract
import numpy as np
from matplotlib.pyplot import ylabel, plot, show
import LoadArchive as LArc
from ConfigDbSing import ConfigDbSing
import pandas as pd
from datetime import datetime, time, timedelta

if __name__=="__main__":
		print(" Test TA-Lib ")
		_connect_db = ConfigDbSing().get_config()
		pref_comp = _connect_db["comp_pref"]

		_data = LArc.LoadArchive.Picle(f"{pref_comp}Trading03\\Data\\Sber\\candles1H.pickle")
		_pd = _data.Pd

		# data=yf.download('SPY', start='2021-01-01', end='2021-12-31')
		_pd['datetime'] = pd.to_datetime(_pd['date'].astype(str) + ' ' + _pd['time'].astype(str))
		_pd.drop(['date', 'time'], axis=1, inplace=True)
		# _pd['datetime'] = pd.Series(_pd['datetime'])
		_pd = _pd.set_index('datetime')
		print(_pd.head())
		data = _pd.copy()
		data.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Adj Close':'close', 'Volume':'volume'}, inplace=True)
		# print(data)
		d=np.array(data['close'])
		sma = talib.SMA(d)

		plot(sma)
		ylabel('---')
		show()

		# mpf.plot(data, type='line')
		mpf.plot(data,type='line',mav=(3,6,9),volume=True)
		# mpf.plot(data,type='candle',mav=(3,6,9),volume=True)
		# mpf.plot(data,type='candle',mav=(7,12))
		# mpf.plot(data)
		k=11

		# https://github.com/matplotlib/mplfinance/wiki /Plotting-Too-Much-Data
