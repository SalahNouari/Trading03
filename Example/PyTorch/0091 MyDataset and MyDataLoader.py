# https://www.youtube.com/watch?v=exaWOE8jvy8&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=9
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import LoadArchive as la
import pandas as pd

class WineDataset(Dataset):

		def __init__(self):
				# Initialize data, download, etc.
				# read with numpy or pandas
				xy = np.loadtxt('./data/wine-red.csv', delimiter=',', dtype=np.float32, skiprows=1)
				self.n_samples = xy.shape[0]

				# here the first column is the class label, the rest are the features
				self.x_data = torch.from_numpy(xy[:, 1:])  # size [n_samples, n_features]
				self.y_data = torch.from_numpy(xy[:, [0]])  # size [n_samples, 1]

		# support indexing such that dataset[i] can be used to get i-th sample
		def __getitem__(self, index):
				return self.x_data[index], self.y_data[index]

		# we can call len(dataset) to return the size
		def __len__(self):
				return self.n_samples

class DbTicker(Dataset):
		def __init__(self, pathfile:str):
				# Initialize data, download, etc.
				# read with numpy or pandas
				_la = la.LoadArchive()
				data = _la.Picle(pathfile)
				d = data.Pd
				xy =d.to_numpy()
				xy=xy[:,2:]
				self.n_samples = xy.shape[0]
				xx = xy[:, :-1].astype(np.float32)
				xx1=torch.from_numpy(xx[:,0])
				# self.x_data = torch.from_.from_numpy(xy[:, :1])  # size [n_samples, n_features]
				# self.y_data = torch.from_numpy(xy[:, [0]])  # size [n_samples, 1]

				j=1

		# support indexing such that dataset[i] can be used to get i-th sample
		def __getitem__(self, index):
				return self.x_data[index], self.y_data[index]

		# we can call len(dataset) to return the size
		def __len__(self):
				return self.n_samples

if __name__ == "__main__":
		dbTicker = DbTicker("E:\Trading03\Data\Sber\candles1H")
		i=1
# create dataset
# dataset = WineDataset()
#
# # get first sample and unpack
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)
#
# # Load whole dataset with DataLoader
# # shuffle: shuffle data, good for training
# # num_workers: faster loading with multiple subprocesses
# # !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
# train_loader = DataLoader(dataset=dataset,
# 																										batch_size=4,
# 																										shuffle=True,
# 																										num_workers=2)
#
# # convert to an iterator and look at one random sample
# # dataiter = iter(train_loader)
# # data = next(dataiter)
# # features, labels = data
# # print(features, labels)

# Dummy Training loop

# num_epochs = 2
# total_samples = len(dataset)
# n_iterations = math.ceil(total_samples / 4)
# print(total_samples, n_iterations)
# for epoch in range(num_epochs):
# 		for i, (inputs, labels) in enumerate(train_loader):
#
# 				# here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
# 				# Run your training process
# 				if (i + 1) % 5 == 0:
# 						print(
# 								f'Epoch: {epoch + 1}/{num_epochs}, Step {i + 1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')
#


l=1