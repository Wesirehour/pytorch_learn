import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Net

train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 长度获取
train_data_len = len(train_data)
test_data_len = len(test_data)
print('训练集长度：{}'.format(train_data_len))
print('测试集长度：{}'.format(test_data_len))

# dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
model = Net()
# if torch.cuda.is_available():
#     model = model.cuda()
model = model.to(device)

# 损失函数
loss = nn.CrossEntropyLoss()
# if torch.cuda.is_available():
#     loss = loss.cuda()
loss = loss.to(device)

# 优化器
learning_rate = 0.01  # 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数
epochs = 10

# tensorboard记录
writer = SummaryWriter('./train_log')

for epoch in range(epochs):
    # 开始训练
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss_result = loss(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss_result.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 50 == 0:
            print('训练次数: {}, loss: {}'.format(total_train_step, loss_result.item()))
            writer.add_scalar('train_loss', loss_result.item(), total_train_step)

    # 开始测试
    model.eval()
    test_loss = 0
    total_acc = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            # if torch.cuda.is_available():
            #     imgs = imgs.cuda()
            #     targets = targets.cuda()
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            acc = (outputs.argmax(1) == targets).sum()
            total_acc += acc
            loss_result = loss(outputs, targets)
            test_loss += loss_result
    total_test_step += 1
    print('测试集loss: {}'.format(test_loss))
    print('测试集整体正确率: {}'.format(total_acc/test_data_len))

    writer.add_scalar('test_loss', test_loss, total_test_step)
    writer.add_scalar('test_acc', total_acc/test_data_len, total_test_step)

    torch.save(model, 'model_{}.pth'.format(epoch))

writer.close()
