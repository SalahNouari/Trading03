# https://neurohive.io/ru/tutorial/cnn-na-pytorch/

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import  matplotlib.pyplot as plt

import time
Lstime = []
device = "cuda" if torch.cuda.is_available() else "cpu"

class  ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7*7*64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out




if __name__ == '__main__':
    print(" ===---  MNIST_conv2D ---=== ")
    print(f"Using ->>  {device}  <--- device")

    training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor(), )
    test_data = datasets.FashionMNIST( root="data", train=False, download=True, transform=ToTensor() )

    batch_size = 64*2
    num_epochs = 5
    classes = 10
    learning_rate = 0.001

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break


    _model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(_model.parameters(), lr=learning_rate)

    total_step = len(train_dataloader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            # Прямой запуск
            images, labels = images.to(device), labels.to(device)
            outputs = _model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Обратное распространение и оптимизатор
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Отслеживание точности
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct/total)
            optimizer.lr = 0.0001

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

    _model.eval()
    with  torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = _model(images)
            _,  predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))



    kkk=1
