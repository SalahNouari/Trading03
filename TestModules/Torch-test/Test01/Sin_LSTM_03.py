
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn.init as init
import  matplotlib.pyplot as plt
import copy
import sklearn
import numpy as np
import pylab as pl
import math
import time

Lstime = []
device = "cuda" if torch.cuda.is_available() else "cpu"

import math

class LSTMTargger(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMTargger, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, future=0, y=None):
        outputs =[]
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)
        for i, input_t in enumerate(input.chunk(input_size(1), dim=1)):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


def trasform_data(data0, ncount):
    x, y = [], []
    for i in range(len(data0)-ncount):
        x_i = data[i:i+ncount]
        y_i =data[i+1:i+ncount+1]
        x.append(x_i)
        y.append(y_i)
    x_arr = np.array(x).reshape(-1, nPoint)
    y_arr = np.array(y).reshape(-1, nPoint)
    x_arr = Variable(torch.from_numpy(x_arr).float())
    y_arr = Variable(torch.from_numpy(y_arr).float())
    return x_arr, y_arr


if __name__ == '__main__':
    print(" Sin_LSTM_03 ")
    dtype = torch.FloatTensor
    input_size, hidden_size, output_size = 7, 6, 1
    epochs = 5
    seq_length = 500
    lr = 0.1
    data_time_stops = np.linspace(2, seq_length, seq_length+1)
    data = np.sin(data_time_stops*2*math.pi*5)*np.sin(data_time_stops*2*math.pi*2)*2
    data.resize((seq_length+1, 1))
    plt.figure()
    plt.plot(data)
    plt.show()

    # x = Variable(torch.Tensor(data[:-1]).type(dtype), requires_grad=False)
    # y = Variable(torch.Tensor(data[1:]).type(dtype), requires_grad=False)
    batch_size = 64
#    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # def calculate_accuracy(y_true, y_pread):
    #     predict = y_pread.ge(.5).view(-1)
    #     return y_true-predict

    model = LSTMTargger(1, 2,4)
    loss_function = nn.NLLLoss()
#    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)



    nPoint = 200
    # x_trainMas = []
    # y_testMas = []
    # _count_x = len(x)
    # for i in range(nPoint, _count_x):
    #     x_trainMas.append(x[i-nPoint:i])
    #     y_testMas.append(y[i-nPoint:i])

# x_train = DataLoader(train, batch_size=batch_size)

#     for i0 in range(len( x_trainMas)):
#         x_train = x_trainMas[i0]
# #        print(x_train.shape)
#         y_train = y_testMas[i0]
# #        print(y_train.view(len(y_train), -1)) #
# #        print(y_train.shape)


    x_train, y_train = trasform_data(data[:-nPoint], nPoint)
    x_test, y_test = trasform_data(data[len(data) : ], nPoint)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    for i in range(epochs):
        print(f" epoch ->  {i+1}")
#        y_pred = model(x_train.view(len(x_train), -1))
        y_pred = model(x_train)
#       print(y_train.view(len(x_train), -1)) #

        y_pred = torch.squeeze(y_pred)
        train_loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        jj=1


#    plt.show()
