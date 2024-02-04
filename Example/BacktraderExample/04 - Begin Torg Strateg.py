# https://www.backtrader.com/docu/quickstart/quickstart/
# https://www.backtrader.com/
# ---  раздел документации  ---
# https://www.backtrader.com/docu/concepts/

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt

import LoadArchive as LArc
from ConfigDbSing import ConfigDbSing


class TestStrategy(bt.Strategy):
  """  Пример торговой стратегии без торговли """
  def log(self, txt, dt=None):
    """ Выводим строки с датой """
    dt =dt or self.datas[0].datetime.date(0)    # Заданная дата или дата текущего бара
    print(f"{dt.isoformat()}, {txt}")

  def __init__(self):
    # super(bt.Strategy, self).__init__(self)
    self.DataOpen = self.datas[0].open
    self.DataHigh = self.datas[0].high
    self.DataLow = self.datas[0].low
    self.DataClose = self.datas[0].close

  def next(self):
    self.log(f"Close = {self.DataClose[0]:.2f}  ")

class TestStrategy1(TestStrategy):
  def next(self):
    self.log(f"Open = {self.DataOpen[0]:.2f}  ")


if __name__=="__main__":
  _connect_db = ConfigDbSing().get_config("dan")
  pref_comp = _connect_db.connect_db["comp_pref"]
  var1 = LArc.LoadArchive.Picle(f"{_connect_db.path_not_git_data}\\Gazp\\candles1day.pickle")
  dataframe =var1.Pd
  data = bt.feeds.PandasData(dataname=var1.Pd,  openinterest=-1)

  cerebro = bt.Cerebro()                  # Инициализаци движка Cerebro - "мозг" на испанском
  cerebro.addstrategy(TestStrategy)
  cerebro.addstrategy(TestStrategy1)
  cerebro.adddata(data)                   # привязка данных
  cerebro.broker.setcash(1000000.00)       #  Стартовый капитал
  print('Стартовый капитал: %.2f' % cerebro.broker.getvalue())
  # Старт стратегии
  cerebro.run()
  # Конечный капитал
  print(f'Конечный капитала: {cerebro.broker.getvalue():.2f}')
  cerebro.plot()

