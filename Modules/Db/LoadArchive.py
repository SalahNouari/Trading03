import os
import pickle


class LoadArchive:
		def __init__(cls):
				pass

		@classmethod
		def Picle(cls, namefile):
				if not (".pickle" in namefile):
						namefile = namefile + ".pickle"

				if os.path.exists(namefile):
						file = open(namefile, 'rb')
						data = pickle.load(file)
						file.close()
						return data

				print(f" - нет файла {namefile}")
				raise " Нет загружаемого файла "
