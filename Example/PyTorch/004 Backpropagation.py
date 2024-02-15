
# https://www.youtube.com/watch?v=exaWOE8jvy8&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=4
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
				"""
		# Forward pass a compute the loss
		x = torch.tensor(1.0)
		y = torch.tensor(2.0)
		w = torch.tensor(1.0, requires_grad=True)
		y_hat = w * x
		loss = (y_hat - y)**2
		print(loss)
		# backward pass
		loss.backward()
		print(w.grad)
		# update weights
		# next Forward and backward


