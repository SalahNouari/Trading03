import os
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from enum import Enum
import pickle

from ConfigDbSing import ConfigDbSing


class TSave(Enum):
		numpy = 1
		pandas = 2
		pickle = 3

		def Get(st):
				switch = {
						'numpy': TSave.numpy,
						'pandas': TSave.pandas,
						'pickle': TSave.pickle,
						'': TSave.pickle
				}
				return switch.get(st)


class LoadCSV:
		class DbBase:
				pass
		def __init__(self, **params):
				"""
					PathDirRead 	- путь к каталогу исходным данных E:\\Trading03\\Data\\CSV\\Gazp',
					PathDirWrite - путь к каталогу записи в каталог E:\\Trading03\\Data\\NotGit\\Gazp',
					FileRead - имя файла (если читаем данные из одного файла) - 'GAZP_19_22.csv',
					TypeData - тип сохраненных данных  pickle, numpy, pandas
					NameInDb - записать название тикира в pickle - 'Gazp',
					NameFileWrite - записать в файл 'candles1day')
				"""
				self._lsPath = []
				self._pd = None
				self._numpy = None
				self.__paramKey = TSave.pickle
				self.__nametime = ""
				self.__dbtime = ""
				_connect_db = ConfigDbSing.getСonfig()
				self._IniciallParams(params)

		def _IniciallParams(self, params):
				self._pathCsvRead = params.get("PathDirRead", "")
				self._pathCsvWrite = params.get("PathDirWrite", "")
				__typesave = params.get("TypeData", "")
				self.__paramKey = TSave.Get(__typesave)
				self.__nameInDb = params.get("NameInDb", "")
				self.__nameFileWrite = params.get("NameFileWrite", "")
				self.__dbTime = ""
				self.__namefileread = params.get("FileRead", "")
				if self.__namefileread != "" and self._pathCsvRead !="":
					self.__pathnameOneFileread = self._pathCsvRead+"\\" + self.__namefileread
				else:
						self.__pathnameOneFileread=""


				if not os.path.exists(self._pathCsvRead):
						print(f" Ошибка нет каталога с исходными данными csv: - {self._pathCsvRead}")
						raise ' ошибка с каталогами '

				print(f" Каталог с исходными данными csv: - {self._pathCsvRead}")
				print(f" Каталог куда пишим данные: - {self._pathCsvWrite}")

				print(f" Тип данных для записи - {str(self.__paramKey.name)}")

		def _saveToFile(self):
				match self.__paramKey:
						case TSave.pickle:
								self.SavePicle()

						case TSave.pandas:
								print("---  pandas  ---- ")

						case TSave.numpy:
								self.SaveNumpy()

						case _:
								self.SavePicle()

		def Run(self, params=None):
				"""
				Можно так же записать данные как в конструктор
				"""
				if None != params:
						self._IniciallParams(params)

				if os.path.isfile(self.__pathnameOneFileread):
						self.readCsvFilePd(self.__pathnameOneFileread)
						self._calcTimeFrame()
						self._saveToFile()
						return

				if not os.path.exists(self._pathCsvWrite):
						os.mkdir(self._pathCsvWrite)

				self._lsPath = [self._pathCsvRead + "\\" + x for x in os.listdir(self._pathCsvRead) if '.csv' in x]
				if self._lsPath.__len__() == 0:
						print(f" Ошибка нет файлов csv в каталоге {self._pathCsvRead}")
						raise 'Нет csv файлов'
				self._lsPath.sort()

				for _path in self._lsPath:
						self.readCsvFilePd(_path)

				self._calcTimeFrame()

				self._saveToFile()


		def _calcTimeFrame(self):
				_dd0 = self._pd['date'].iloc[0]
				_dd1 = self._pd['date'].iloc[1]
				_dt0 = self._pd['time'].iloc[0]
				_dt1 = self._pd['time'].iloc[1]
				_dt0 = datetime(_dd0.year, _dd0.month, _dd0.day, _dt0.hour, _dt0.minute, _dt0.second)
				_dt1 = datetime(_dd1.year, _dd1.month, _dd1.day, _dt1.hour, _dt1.minute, _dt1.second)

				time_delta = _dt1 - _dt0
				_date = time_delta.days
				_tine = time_delta.seconds
				if _date>0:
						if _date>7:
								self.__dbtime = "month"
								return
						elif _date == 7:
								self.__dbtime = "week"
						elif _date<6:
								_numDay = "" if _date==1 else str(_date)
								self.__dbtime = _numDay+"day"
				if _tine>0:
						_minute = _tine//60
						_hour=_minute//60
						if _hour>0:
								self.__dbtime = str(_hour) + "hour"
						else:
								self.__dbtime = str(_minute) + "min"

				# 'datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest'

				self._pd['datetime'] = pd.to_datetime(self._pd['date'].astype(str) + ' ' + self._pd['time'].astype(str))
				self._pd.drop(['date', 'time'], axis=1, inplace=True)
				# _pd['datetime'] = pd.Series(_pd['datetime'])
				self._pd = self._pd.set_index('datetime')
				# self._pd.drop(['datetime'], axis=1, inplace=True)
				print(f" Рабочий TimeFrame {self.__dbtime}")
				print(self._pd.head())

		def readCsvFilePd(self, _path):
				_tr0 = pd.read_csv(_path, sep=";")
				print(f" Данные из файла {_path} прочитанны ")

				_tr0["<DATE>"] = pd.to_datetime(_tr0["<DATE>"].astype(str), yearfirst=True).dt.date
				_tr0["<TIME>"] = pd.to_datetime(_tr0["<TIME>"], format='%H:%M:%S').dt.time

				_tr0.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
				self._pd = pd.concat([self._pd, _tr0], ignore_index=True)

		def SavePicle(self):
				"""
					Записываем в файл => pathwrite \\ nametime.pickle
				"""
				file = open(self._pathCsvWrite+"\\" + self.__nameFileWrite + ".pickle", 'wb')
				data = LoadCSV.DbBase()
				data.Name = self.__nameInDb
				data.Pd = self._pd
				data.DbTime = self.__dbtime
				pickle.dump(data, file)
				file.close()

		def SaveNumpy(self):
				var00 = self._pd.to_numpy()
				# file = open(self._pathCsvWrite+"\\"+self.__nametime+".npz", 'wb')
				# file = self._pathCsvWrite+"\\"+self.__nametime+".npz"
				file = self._pathCsvWrite+"\\"+self.__nametime+".npz"
				# np.savez('data.npz', arr1=arr1, arr2=arr2)

				arr1 = np.array([1, 2, 3, 4, 5])
				arr2 = np.array([6, 7, 8, 9, 10])
				np.savez(file, arr1=arr1, arr2=arr2)

				# np.save(file, var00)
				kk=1
