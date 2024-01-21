# https://www.backtrader.com/docu/quickstart/quickstart/
# https://www.backtrader.com/
# ---  раздел документации  ---
# https://www.backtrader.com/docu/concepts/
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt

if __name__=="__main__":
    cerebro = bt.Cerebro()                  # Инициализаци движка Cerebro - "мозг" на испанском
    #  Стартовый капитал
    cerebro.broker.setcash(1000000.0)
    print('Стартовый капитал: %.2f' % cerebro.broker.getvalue())
    # Старт стратегии
    cerebro.run()
    # Конечный капитал
    print(f'Конечный капитала: {cerebro.broker.getvalue():.2f}')