
import LoadCSV as LCSV
from ConfigDbSing import ConfigDbSing
import LoadArchive as LArc


if __name__=="__main__":
		print(" Загрузка class для чтения данных CSV ")

		_connect_db = ConfigDbSing().get_config()
		pref_comp = _connect_db["comp_pref"]

		# _lcsv = LCSV.LoadCSV(pathread = f'{pref_comp}Data\\Traiding\\CSV\\Sber\\1Hour',
		# 																						pathwrite = f'{pref_comp}Trading03\\Data\\Sber',
		# 																						typedata = "pickle",
		# 																						name = 'Sber',
		# 																						nametime = 'candles1H')
		#
		# _lcsv.Run()

		var1 = LArc.LoadArchive.Picle(f"{pref_comp}Trading03\\Data\\Sber\\candles1H.pickle")
		k=1

