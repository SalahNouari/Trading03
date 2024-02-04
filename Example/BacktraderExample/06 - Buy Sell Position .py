# https://www.backtrader.com/docu/quickstart/quickstart/
# https://www.backtrader.com/
# ---  раздел документации  ---
# https://www.backtrader.com/docu/concepts/

"""
Пример
https://www.youtube.com/watch?v=CS539ZFiomo&list=PLjEwgdnQO09V7Oj-0-Nc9a7ihI0kaq577&index=10
"""

from __future__ import (absolute_import, division, print_function,
																								unicode_literals)

import backtrader as bt

import LoadArchive as LArc
from ConfigDbSing import ConfigDbSing


class TestStrategy(bt.Strategy):
		"""  Пример торговой стратегии без торговли """

		def log(self, txt, dt=None):
				""" Выводим строки с датой """
				dt = dt or self.datas[0].datetime.date(0)  # Заданная дата или дата текущего бара
				print(f"{dt.isoformat()}, {txt}")

		def __init__(self):
				# super(bt.Strategy, self).__init__(self)
				self.DataClose = self.datas[0].close
				self.Order = None

		def notify_order(self, order):
				""" Изменение статуса заявок """
				if order.status in [order.Submitted, order.Accepted]:  # заявка не исполнена - отправлена или принята брокуром
						return  # статус заявки не изменилась

				if order.status in [order.Completed]:
						if order.isbuy():  # заявка на покупку
								self.log(f'Buy @{order.executed.price:.2f}')
						elif order.issell():  # заявка на продажу
								self.log(f'Sell @{order.executed.price:.2f}')
						self.BarExecuted = len(self)  # Номер бара, на котором была исполнена заявка
				elif order.status in [order.Canceled, order.Margin, order.Rejected]:
						self.log("Canceled/Margin/Rejected")
				self.Order = None

		def next(self):
				''' Приход нового бара '''
				self.log(f'Close = {self.DataClose[0]:.2f}  ')
				if self.Order:  # Если есть неиспользованная заявка
						return  # то выходим дальше не продолжаем

				if not self.position:
						isSignalBuy = self.DataClose[0] < self.DataClose[-1] and self.DataClose[-1] < self.DataClose[-2]

						if isSignalBuy:
								self.log('Buy Marcket')
								self.Order = self.buy()
				else:
						isSignalSell = len(self)-self.BarExecuted>=5 # Прошло не мение 5-и баров с моментом входа в позицию
						if isSignalSell:
								self.log('Sell Market')
								self.Order = self.sell()


if __name__ == "__main__":
		_connect_db = ConfigDbSing().get_config("dan")
		pref_comp = _connect_db.connect_db["comp_pref"]
		var1 = LArc.LoadArchive.Picle(f"{_connect_db.path_not_git_data}\\Gazp\\candles1day.pickle")
		dataframe = var1.Pd
		data = bt.feeds.PandasData(dataname=var1.Pd, openinterest=-1)

		cerebro = bt.Cerebro()  # Инициализаци движка Cerebro - "мозг" на испанском
		cerebro.addstrategy(TestStrategy)
		cerebro.adddata(data)  # привязка данных
		cerebro.broker.setcash(1000000.00)  # Стартовый капитал
		print('Стартовый капитал: %.2f' % cerebro.broker.getvalue())
		# Старт стратегии
		cerebro.run()
		# Конечный капитал
		print(f'Конечный капитала: {cerebro.broker.getvalue():.2f}')
		cerebro.plot()

