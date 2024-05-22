import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(weights=None, progress=True)  # 不预训练
vgg16_true = torchvision.models.vgg16(weights='DEFAULT', progress=True)  # 预训练
# print(vgg16_true)
# VGG16 是1000分类的模型，需要对它修改，以适应10分类的任务
dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)

# 增加1000->10的线性层
# vgg16_true.classifier.add_module('add_linear', nn.Linear(1000,10))
# print(vgg16_true)

# 修改最后一层线性层的out_features=10
vgg16_false.classifier[6]=nn.Linear(in_features=4096, out_features=10, bias=True)
print(vgg16_false)
