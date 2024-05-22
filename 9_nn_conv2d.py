import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


net = Net()
# print(net)

writer = SummaryWriter('./nn_conv2d_logs')
step = 1
for data in dataloader:
    imgs, targets = data
    output = net(imgs)
    # torch.Size([64, 3, 32, 32])
    # print(imgs.shape)
    # torch.Size([64, 6, 30, 30])
    # print(output.shape)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images('input', imgs, step)
    writer.add_images('output', output, step)
    step += 1

writer.close()
