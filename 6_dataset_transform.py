import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

dataset_transform = transforms.Compose([
    transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=dataset_transform, download=True)

# print(test_set[0])
# img, target = test_set[0]
# img.show()
# print(test_set.classes[target])

writer = SummaryWriter('dataset_transform_logs')
for i in range(10):
    img, target = test_set[i]
    writer.add_image('test', img, i)

writer.close()