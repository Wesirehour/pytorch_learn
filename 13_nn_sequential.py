import torch
from torch import nn
from torch.nn import MaxPool2d, Conv2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = Conv2d(3, 32, 5, padding=2)
		self.maxpool1 = MaxPool2d(2)
		self.conv2 = Conv2d(32, 32, 5, padding=2)
		self.maxpool2 = MaxPool2d(2)
		self.conv3 = Conv2d(32, 64, 5, padding=2)
		self.maxpool3 = MaxPool2d(2)
		self.flatten = Flatten()
		self.linear1 = Linear(1024, 64)
		self.linear2 = Linear(64, 10)

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


net = Net()
writer = SummaryWriter("nn_sequential_logs")
writer.add_graph(net, torch.ones(64, 3, 32, 32))
writer.close()
