import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d, Conv2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model1 = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
net = Net()
optimer = torch.optim.SGD(net.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        inputs, targets = data
        outputs = net(inputs)
        result_loss = loss(outputs, targets)
        optimer.zero_grad()
        result_loss.backward()
        optimer.step()
        # print(result_loss)
        running_loss += result_loss.item()
    print(running_loss / len(dataloader))
