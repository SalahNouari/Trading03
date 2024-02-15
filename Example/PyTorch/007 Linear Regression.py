# https://www.youtube.com/watch?v=exaWOE8jvy8&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=7
# Учимся работвть с тензером

import torch
import torch.nn as nn
import numpy as np
from numpy import random
from sklearn import datasets
import matplotlib.pyplot as plt


class LinearRegression(nn.Module):
		def __init__(self, input_dim, output_dim):
				super(LinearRegression, self).__init__()
				# define diferent layers
				self.lin = nn.Linear(input_dim, output_dim)

		def forward(self, x):
				return self.lin(x)


if __name__ == "__main__":
		print(torch.__version__)
		print(torch.cuda.is_available())
		if torch.cuda.is_available():
				device = torch.device("cuda")
		else:
				device = torch.device("cpu")
		print(device)

		# 0. get-> set data
		X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
		X = torch.from_numpy(X_numpy.astype(np.float32))
		y = torch.from_numpy(Y_numpy.astype(np.float32))
		y = y.view(y.shape[0], 1)
		n_sample, n_features = X.shape

		# 1. model
		input_size = n_features
		output_size = 1
		model = nn.Linear(input_size, output_size)

		# 2. loss and optimizer
		learing_rate = 0.001
		criterion = nn.MSELoss()
		optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate)

		# 3. training loop
		num_epochs = 10000
		for epoch in range(num_epochs):
				# predict = forward pass with our model
				y_predicted = model(X)
				loss = criterion(y_predicted, y)

				# backward pass
				loss.backward()

				# update
				optimizer.step()

				optimizer.zero_grad()

				# print(f"epocha: {epoch + 1}, loss = {loss.item():.4f}  ")

				if(epoch+1)%10 == 0:
						print(f"epocha: {epoch+1}, loss = {loss.item():.4f}  ")

		# plot
		predicted = model(X).detach().numpy()
		plt.plot(X_numpy, Y_numpy, "ro")
		plt.plot(X_numpy, predicted, "b")
		plt.show()

