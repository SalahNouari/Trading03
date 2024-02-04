
import LoadCSV as LCSV
from ConfigDbSing import ConfigDbSing
import LoadArchive as LArc


if __name__=="__main__":
		print(" Загрузка class для чтения данных CSV ")

		_connect_db = ConfigDbSing().get_config("dan")
		pref_comp = _connect_db.connect_db["comp_pref"]

		_lcsv = LCSV.LoadCSV(PathDirRead = f'{pref_comp}Trading03\\Data\\CSV\\Gazp',
																								PathDirWrite = f'{_connect_db.path_not_git_data}\\Gazp',
																							 FileRead = 'GAZP_19_22.csv',
																								TypeData = "pickle",
																								NameInDb = 'Gazp',
																								NameFileWrite = 'candles1day')

		_lcsv.Run()

		var1 = LArc.LoadArchive.Picle(f"{_connect_db.path_not_git_data}\\Gazp\\candles1day.pickle")
		k=1

		'''
				_lcsv = LCSV.LoadCSV(pathread = f'{pref_comp}Trading03\\Data\\CSV\\Sber\\1Hour',
																										pathwrite = f'{_connect_db.path_not_git_data}\\Sber',
																										typedata = "pickle",
																										name = 'Sber',
																										nametime = 'candles1H')
		
		'''