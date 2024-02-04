# https://www.backtrader.com/docu/quickstart/quickstart/
# https://www.backtrader.com/
# ---  раздел документации  ---
# https://www.backtrader.com/docu/concepts/

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt

import LoadArchive as LArc
from ConfigDbSing import ConfigDbSing

if __name__=="__main__":
  _connect_db = ConfigDbSing().get_config("dan")
  pref_comp = _connect_db.connect_db["comp_pref"]

  var1 = LArc.LoadArchive.Picle(f"{_connect_db.path_not_git_data}\\Gazp\\candles1day.pickle")
  dataframe =var1.Pd
  data = bt.feeds.PandasData(dataname=dataframe,  openinterest=-1
                             # ,fromdate=datetime.datetime(2019, 1, 1)
                             # ,todate=datetime.datetime(2020, 12, 31)
                             )

  cerebro = bt.Cerebro()                  # Инициализаци движка Cerebro - "мозг" на испанском

  cerebro.adddata(data)                   # привязка данных
  cerebro.broker.setcash(1000000.0)       #  Стартовый капитал
  print('Стартовый капитал: %.2f' % cerebro.broker.getvalue())
  # Старт стратегии
  cerebro.run()
  # Конечный капитал
  print(f'Конечный капитала: {cerebro.broker.getvalue():.2f}')
  cerebro.plot()
'''
    data = bt.feeds.GenericCSVData( # Можно загрузить любой CSV файл в соответствующем формате
    fromdate = datetime.datetime(2019, 1, 1),
    todate = datetime.datetime(2022, 12, 31),
    date=0,
    time=1,
    open=2,
    high=3,
    low=4,
    close=5,
    volume=6,
    openinterest=-1
    )

'''
