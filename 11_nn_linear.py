import torch
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(196608, 10)

    def forward(self, x):
        x = self.linear1(x)
        return x

net = Net()
for data in dataloader:
    imgs, targets = data
    # print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs)
    # print(output.shape)
    output = net(output)
    print(output.shape)
