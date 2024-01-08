import json
import os
import socket


class ConfigDbSing:
		__instance = None
		connect_db = None
		path_basa = ""
		path_db = ""
		path_not_git_data = ""
		path_json = ''
		path_notgit = ''
		name_file_json = r'config_db.json'

		# path - путь для файла file.json

		def __init__(self, name_comp_path=None, path=None):
				self._initializationData(name_comp_path, path)

		def _initializationData(self, name_comp_path=None, path=None):
				print("!!!!!!!!!!  ConfigDbSing ")
				name_comp = (socket.gethostname()).lower()
				if name_comp_path is None:
						name_comp_path = {'p1': "E:\\", "ws778": "E:\\MLserver\\"}

				try:
						pref_comp = name_comp_path[name_comp]
				except:
						raise " Нет заложенных путей "

				if path is None:
						path = "Trading03\\Modules"
				# raise ' Нужно установить путь к конфигурации '

				''' проверка существования основной директории  '''
				ConfigDbSing.path_basa = pref_comp + path
				if not os.path.isdir(ConfigDbSing.path_basa):
						raise ' Нет директории  '

				ConfigDbSing.path_db = ConfigDbSing.path_basa + "\\DB"
				if not (os.path.isdir(ConfigDbSing.path_db)):
						raise ' Нет директории DB '

				ConfigDbSing.path_notgit = ConfigDbSing.path_basa + "\\NotGit"
				if not (os.path.isdir(ConfigDbSing.path_notgit)):
						os.mkdir(ConfigDbSing.path_notgit)

				ConfigDbSing.path_not_git_data = ConfigDbSing.path_notgit + "\\Data"
				if not (os.path.isdir(ConfigDbSing.path_not_git_data)):
						os.mkdir(ConfigDbSing.path_not_git_data)

				ConfigDbSing.path_json = ConfigDbSing.path_db + "\\" + ConfigDbSing.name_file_json
				if not (os.path.exists(ConfigDbSing.path_json)):
						raise ' Нет файла  config_db.json'

				# ConfigDbSing.set_config(self)

				with open(ConfigDbSing.path_json, 'r') as j:
						print(" Читаем данные из файла")
						ConfigDbSing.connect_db = json.load(j)

				# Name DB -----------------------------------------------
				ConfigDbSing.connect_db.update(dbname="DbTrade")
				ConfigDbSing.connect_db.update(comp_pref=pref_comp)
				print(ConfigDbSing.connect_db)
				ConfigDbSing.set_config(self)

		@classmethod
		def getInstance(cls):
				if not cls.__instance:
						cls.__instance = ConfigDbSing
				return cls.__instance

		@classmethod
		def getСonfig(cls):
				return ConfigDbSing.connect_db

		def get_config(self):
				if ConfigDbSing.path_basa.__len__() == 0:
						self._initializationData()
				return ConfigDbSing.connect_db

		def set_config(cls):
				with open(ConfigDbSing.path_json, 'w') as file:
						json.dump(ConfigDbSing.connect_db, file)

		def path_dan(self):
				return ConfigDbSing.connect_db['Data']

		def path_files(self, s):
				return ConfigDbSing.connect_db['comp_pref'] + s


"""

 def get_mane_comp():
  return (socket.gethostname()).lower()
      if name_comp == 'p1':
        pref_comp = "E:\\"
      else:
        pref_comp = "E:\MLserver\\"
"""
