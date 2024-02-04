# https://www.backtrader.com/docu/quickstart/quickstart/
# https://www.backtrader.com/
# ---  раздел документации  ---
# https://www.backtrader.com/docu/concepts/

"""
Пример
https://www.youtube.com/watch?v=k0QR-ymDMR4&list=PLjEwgdnQO09V7Oj-0-Nc9a7ihI0kaq577&index=12
"""

from __future__ import (absolute_import, division, print_function,
																								unicode_literals)

import backtrader as bt

import LoadArchive as LArc
from ConfigDbSing import ConfigDbSing


class TestStrategy(bt.Strategy):
		"""  Пример торговой стратегии без торговли """
		params = ( # Параметр торговой системы
				('SMAPeriod', 26), # Период SMA
		)
		def log(self, txt, dt=None):
				""" Выводим строки с датой """
				dt = dt or self.datas[0].datetime.date(0)  # Заданная дата или дата текущего бара
				print(f"{dt.isoformat()}, {txt}")

		def __init__(self):
				# super(bt.Strategy, self).__init__(self)
				self.BarExecuted = None
				self.DataClose = self.datas[0].close
				self.Order = None
				self.sma = bt.indicators.SMA(self.datas[0], period=self.params.SMAPeriod)

		def notify_order(self, order):
				""" Изменение статуса заявок """
				if order.status in [order.Submitted, order.Accepted]:  # заявка не исполнена - отправлена или принята брокуром
						return  # статус заявки не изменилась

				if order.status in [order.Completed]:
						if order.isbuy():  # заявка на покупку
								self.log(f'Buy @{order.executed.price:.2f}, Cost={order.executed.value:.2f}, Comm={order.executed.comm:.2f}')
						elif order.issell():  # заявка на продажу
								self.log(f'Sell @{order.executed.price:.2f}, Cost={order.executed.value:.2f}, Comm={order.executed.comm:.2f}')
						self.BarExecuted = len(self)  # Номер бара, на котором была исполнена заявка
				elif order.status in [order.Canceled, order.Margin, order.Rejected]:
						self.log("Canceled/Margin/Rejected")
				self.Order = None

		def notify_trade(self, trade):
				""" Изменение статуса позиции """
				if not trade.isclosed:		# Если позиция не закрыта
						return 			# статус позиции не изменен

				self.log(f'Trade Profit, Gross={trade.pnl:.2f}, NET={trade.pnlcomm:.2f}')

		def next(self):
				""" Приход нового бара """
				self.log(f'Close = {self.DataClose[0]:.2f}  ')
				if self.Order:  # Если есть неиспользованная заявка
						return  # то выходим дальше не продолжаем

				if not self.position:
						isSignalBuy = self.DataClose[0] > self.sma[0]
						if isSignalBuy:
								self.log('Buy Marcket')
								self.Order = self.buy()
				else:
						isSignalSell = self.DataClose[0]<self.sma[0]
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
		cerebro.addstrategy(TestStrategy, SMAPeriod=15)
		cerebro.adddata(data)  # привязка данных
		cerebro.broker.setcash(1000000.00)  # Стартовый капитал
		cerebro.broker.setcommission(commission=0.001)		# коммисия от брокера 0.1% от суммы каждой сделки
		cerebro.addsizer(bt.sizers.FixedSize, stake=10)
		cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='TradeAnalyzer')
		print('Стартовый капитал: %.2f' % cerebro.broker.getvalue())
		# Старт стратегии
		result = cerebro.run()
		# Конечный капитал
		print(f'Конечный капитала: {cerebro.broker.getvalue():.2f}')
		analysis = result[0].analyzers.TradeAnalyzer.get_analysis()
		print(' Прибыль/убытки по закрытым сделкам ')
		print(f'- Без комиссии {analysis["pnl"]["gross"]["total"]:.2f}')
		print(f'- C комиссией  {analysis["pnl"]["net"]["total"]:.2f}')
		# cerebro.plot()
