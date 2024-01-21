# https://www.backtrader.com/docu/quickstart/quickstart/
# https://www.backtrader.com/
# ---  раздел документации  ---
# https://www.backtrader.com/docu/concepts/

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import backtrader as bt
import pandas as pd
import numpy as np


if __name__=="__main__":
    cerebro = bt.Cerebro()                  # Инициализаци движка Cerebro - "мозг" на испанском
    '''
    data = bt.feeds.GenericCSVData( # Можно загрузить любой CSV файл в соответствующем формате
        dataname = 'GAZP_19_22.csv',
        separator =',',
        dtformat='%Y%m%d',
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
    data = bt.feeds.
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
        dataname = 'GAZP_19_21.csv',
        separator =';',
        dtformat='%Y%m%d',
        formdate=datetime.datetime(2019, 1,1),  # Начальная дата исторических данных
        todate=datetime.datetime(2021, 12, 31)
    )


'''