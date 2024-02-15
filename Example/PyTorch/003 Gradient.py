
# https://www.youtube.com/watch?v=exaWOE8jvy8&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=3
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
		x = torch.rand(3, device=device, requires_grad=True)
		print(x)
		y = x+2
		print(y)
		z=y*y*2
		# z = z.mean() # - убрали для теста
		# print(z)
		# z.backward()  # dz/dx

		v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32, device=device)
		z.backward(v)  # dz/dx
		print(x.grad)
		"""

		""""
				# вторая часть 
		x = torch.rand(3, device=device, requires_grad=True)
		print(x)
		# - отсоединение градиетв
		# 1. x.requires_grad_(False)
		# 2. y = x.detach()
		# 3. with torch.no_grad():
		# 					y=x+2
		# 					print(y)
		"""
		""""
				# третья часть 
		weights = torch.ones(4, requires_grad=True)
		for epoch in range(3):
				model_output = (weights*3).sum()
				model_output.backward()
				print(weights.grad)
				weights.grad.zero_()
		"""

		# не работает 
		weights = torch.ones(4, requires_grad=True)
		optimizer = torch.optim.SGD(weights, lr=0.01)
		optimizer.step()
		optimizer.zero_grad()

