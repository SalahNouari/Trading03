
# https://www.youtube.com/watch?v=exaWOE8jvy8&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=2
# Учимся работвть с тензером

import torch
import numpy as np
from numpy import random

if __name__ == "__main__":
		pass
		"""
		x=torch.rand(4,6, dtype=torch.float64)
		print(x, x.dtype)
		print(x.size())
		"""
		"""
		x=torch.tensor([4.2, 6.3, 4.5])
		print(x, x.dtype)
		print(x.size())

		x = torch.rand(2,2)
		y = torch.rand(2,2)
		print(x)
		print(y)

		z =x + y
		z1= torch.add(x,y)
		print(z)
		print(z1)

		x = torch.rand(2,2)
		print(" x ~~", x)
		y = torch.rand(2,2)
		print(" y ~~",y)
		y.add_(x)
		print(" y1 ~~",y)
		"""
		"""
		x =torch.rand(5, 3)
		print(x)
		print(x[1,:])
		print(x[1,1])
		print(x[1,1].item())
		"""
		"""
		x =torch.rand(4, 4)
		print(x)
		print(x.view(16))
		print(x.view(8,2))
		print(x.view(-1,8))
		"""
		"""
		x = torch.rand(5)
		print(x)
		print(x.numpy())

		a = random.rand(5)
		b =torch.from_numpy(a)
		print(b, type(b))
		"""

		print(torch.__version__)
		print(torch.cuda.is_available())
		z0 = torch.cuda.device(0)
		print(z0)

		if torch.cuda.is_available():
				device = torch.device("cuda")
				x = torch.ones(5, device= device)
				y = torch.ones(5)
				y= y.to(device)
				z = x + y
				z0 = z.to("cpu")
				print(x, y, z, z0)
		else:
				print("not cuda")

