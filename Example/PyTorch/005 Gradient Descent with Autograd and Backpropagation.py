
# https://www.youtube.com/watch?v=exaWOE8jvy8&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=5
# Учимся работвть с тензером

import torch

import numpy as np
from numpy import random

if __name__ == "__main__":
		print(torch.__version__)
		print(torch.cuda.is_available())
		z0 = torch.cuda.device(0)
		print(z0)
		if torch.cuda.is_available():
				device = torch.device("cuda")
		else:
				device = torch.device("cpu")

		"""
		# первая часть 
		# Линейная регрессия !!
		# f = w * x => 		f = 2 * x
		X = np.array([1,2,3,4], dtype=np.float32)
		Y = np.array([2,4,6,8], dtype=np.float32)
		w = 0

		# model prediction
		def forward(x):
				return w*x

		# loss
		def loss(y, y_prediction):
				return ((y - y_prediction)**2).mean()

		# gradient
		# MSE = 1/N*(w*x-1)**2
		# dJ/dw= 1/N 2x (w*x -y)
		def gradient(x,y,y_prediction):
				return np.dot(2*x, y_prediction-y).mean()

		print(f"Прогноз перед тренировкой  f(5) ={forward(5):.3f} ")
		lr=0.001
		n_iters = 200
		for epoch in range(n_iters):
			# model	prediction
			# Forward pass
			y_pred = forward(X)

			# loss
			l = loss(Y, y_pred)

			# gradients
			dw = gradient(X, Y, y_pred)

			# update weights
			w -= lr * dw

			if epoch % 5 == 0:
					print(f" epoch {epoch+1}: w={w:.3f} loss = {l:.8f}")

		print(f"Прогноз перед тренировкой  f(5) ={forward(5):.3f} ")
		"""

		"""
				# вторая часть
		"""
		# Линейная регрессия !!
		# f = w * x => 		f = 2 * x
		X = torch.tensor([1,2,3,4], dtype=torch.float64, device=device)
		Y = torch.tensor([2,4,6,8], dtype=torch.float64, device=device)
		w = torch.tensor(0, dtype=torch.float64, device=device, requires_grad=True)


		# model prediction
		def forward(x):
				return w*x

		# loss
		def loss(y, y_prediction):
				return ((y - y_prediction)**2).mean()

		print(f"Прогноз перед тренировкой  f(5) ={forward(5):.3f} ")
		lr=0.01
		n_iters = 20000
		for epoch in range(n_iters):
			# model	prediction
			# Forward pass
			y_pred = forward(X)

			# loss
			l = loss(Y, y_pred)

			# gradients = backward pass
			l.backward()  # dl/dw

			# update weights
			with torch.no_grad():
				w -= lr * w.grad

			# Zero gradients
			w.grad.zero_()



			if epoch % 5 == 0 :
					print(f" epoch {epoch+1}: w={w:.3f} loss = {l:.8f}")

			if l<lr:
					break
		print(f"Прогноз перед тренировкой  f(5) ={forward(5):.3f} ")

