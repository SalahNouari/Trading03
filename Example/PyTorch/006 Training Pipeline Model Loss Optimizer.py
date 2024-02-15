
# https://www.youtube.com/watch?v=exaWOE8jvy8&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=5
# Учимся работвть с тензером

import torch
import torch.nn as nn


class LinearRegression(nn.Module):
		def __init__(self, input_dim, output_dim):
				super(LinearRegression, self).__init__()
				# define diferent layers
				self.lin = nn.Linear(input_dim, output_dim)

		def forward(self, x):
				return self.lin(x)


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

    '''
    Три этапа обучения в PyTorch
      1.  Desinger model (input? output size, forward pass )
      2.  Construct loss and options
      3.  Training loop   
          - forward pass - вычисляем прогноз
          -  backward pass: gradients  -   шаг назад
          -  update weights - обновляем веса
    '''

    """
    # первая часть 
    # Линейная регрессия !!

    # Линейная регрессия !!
    # f = w * x => 		f = 2 * x
    X = torch.tensor([1,2,3,4], dtype=torch.float64, device=device)
    Y = torch.tensor([2,4,6,8], dtype=torch.float64, device=device)
    w = torch.tensor(0, dtype=torch.float64, device=device, requires_grad=True)


    # model prediction
    def forward(x):
        return w*x

    print(f"Прогноз перед тренировкой  f(5) ={forward(5):.3f} ")

    lr=0.01
    n_iters = 200
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD([w], lr=lr)

    for epoch in range(n_iters):
      # model	prediction
      # Forward pass
      y_pred = forward(X)

      # loss
      l = loss(Y, y_pred)

      # gradients = backward pass
      l.backward()  # dl/dw

      # update weights
      optimizer.step()

      # Zero gradients
      optimizer.zero_grad()



      if epoch % 5 == 0 :
          print(f" epoch {epoch+1}: w={w:.3f} loss = {l:.8f}")

      if l<0.00000001:
          break
    print(f"Прогноз перед тренировкой  f(5) ={forward(5):.3f} ")

    """
    """
        # вторая часть
    """
    # 1) Design model (input, output, forward pass with different layers)
    # 2) Construct loss and optimizer
    # 3) Training loop
    #       - Forward = compute prediction and loss
    #       - Backward = compute gradients
    #       - Update weights



# Linear regression
    # f = w * x

    # here : f = 2 * x

    # 0) Training samples, watch the shape!
    X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
    Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

    n_samples, n_features = X.shape
    print(f'#samples: {n_samples}, #features: {n_features}')
    # 0) create a test sample
    X_test = torch.tensor([5], dtype=torch.float32)

    # 1) Design Model, the model has to implement the forward pass!
    # Here we can use a built-in model from PyTorch
    input_size = n_features
    output_size = n_features

    # we can call this model with samples X
    # model = nn.Linear(input_size, output_size)

    model = LinearRegression(input_size, output_size)

    print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

    # 2) Define loss and optimizer
    learning_rate = 0.001
    n_iters = 1000

    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 3) Training loop
    for epoch in range(n_iters):
        # predict = forward pass with our model
        y_predicted = model(X)

        # loss
        l = loss(Y, y_predicted)

        # calculate gradients = backward pass
        l.backward()

        # update weights
        optimizer.step()

        # zero the gradients after updating
        optimizer.zero_grad()

        if epoch % 10 == 0:
            [w, b] = model.parameters()  # unpack parameters
            print('epoch ', epoch + 1, ': w = ', w[0][0].item(), ' loss = ', l)

    print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')