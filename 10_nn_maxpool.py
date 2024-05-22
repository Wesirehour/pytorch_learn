import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 池化能保留特征减少参数

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        x = self.maxpool1(x)
        return x


net = Net()

step = 1
writer = SummaryWriter('nn_maxpool_logs')
for data in dataloader:
    img, target = data
    writer.add_images('input', img, step)
    output = net(img)
    writer.add_images('output', output, step)
    step += 1

writer.close()