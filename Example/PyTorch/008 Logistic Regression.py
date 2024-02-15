# https://www.youtube.com/watch?v=exaWOE8jvy8&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=7
# Учимся работвть с тензером

import torch
import torch.nn as nn
import numpy as np
from numpy import random
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


class LogisticRegression(nn.Module):
		def __init__(self, n_input_features):
				super(LogisticRegression, self).__init__()
				# define diferent layers
				self.linear = nn.Linear(n_input_features, 1)

		def forward(self, x):
				y_predicted = torch.sigmoid(self.linear(x))
				return y_predicted


if __name__ == "__main__":
		print(torch.__version__)
		print(torch.cuda.is_available())
		if torch.cuda.is_available():
				device = torch.device("cuda")
		else:
				device = torch.device("cpu")
		print(device)

		# 0. prepare data
		bc = datasets.load_breast_cancer()
		X, y = bc.data, bc.target
		n_sample, n_features = X.shape
		print(X.shape)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

		# scale
		sc = StandardScaler()
		X_train = sc.fit_transform(X_train)
		X_test = sc.transform(X_test)

		X_train = torch.from_numpy(X_train.astype(np.float32))
		X_test = torch.from_numpy(X_test.astype(np.float32))
		y_train = torch.from_numpy(y_train.astype(np.float32))
		y_test = torch.from_numpy(y_test.astype(np.float32))

		y_train = y_train.view(y_train.shape[0], 1)
		y_test = y_test.view(y_test.shape[0], 1)

		# 1. model
		# f = wx + b, sigmoid st the end
		model = LogisticRegression(n_features)

		# 2. loss and optimizer
		learing_rate = 0.001
		criterion = nn.BCELoss()
		optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate)

		# 3. training loop
		num_epochs = 1000
		for epoch in range(num_epochs):
				# predict = forward pass with our model
				y_predicted = model(X_train)
				loss = criterion(y_predicted, y_train)

				# backward pass
				loss.backward()

				# update
				optimizer.step()

				optimizer.zero_grad()

				if(epoch+1)%10 == 0:
						print(f"epocha: {epoch+1}, loss = {loss.item():.4f}  ")

		with torch.no_grad():
				y_predicted = model(X_test)
				y_predicted_cls =  y_predicted.round()
				acc = y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
				print(f" accuracy = {acc:.4f} ")
		# # plot
		# predicted = model(X).detach().numpy()
		# plt.plot(X_numpy, Y_numpy, "ro")
		# plt.plot(X_numpy, predicted, "b")
		# plt.show()

