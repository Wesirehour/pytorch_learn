import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False,transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.relu1 = torch.nn.ReLU(inplace=False)
        self.sigmoid1 = torch.nn.Sigmoid()

    def forward(self, x):
        # x = self.relu1(x)
        x = self.sigmoid1(x)
        return x

net = Net()

writer = SummaryWriter('./nn_relu_logs')
step = 1
for data in dataloader:
    img, target = data
    writer.add_images('input', img, step)
    output = net(img)
    writer.add_images('output', output, step)
    step +=1

writer.close()