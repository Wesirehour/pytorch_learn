import torchvision
import torch

vgg16 = torchvision.models.vgg16(weights='imagenet')

# 保存方式1: 模型结构+模型参数
torch.save(vgg16, './model/vgg16_1.pth')
# 加载方式1
model1 = torch.load('./model/vgg16_1.pth')

# 保存方式2: 模型参数
torch.save(vgg16.state_dict(), './model/vgg16_2.pth')
# 加载方式2
model2 = torch.load('./model/vgg16_2.pth')  # 字典
vgg16_2 = torchvision.models.vgg16(weights=None)
vgg16_2.load_state_dict(model2)


# 陷阱: 另一个文件加载模型的时候，会出现缺少Net类的错误。
# 方法: Import引入
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 5)

    def forward(self, x):
        x = self.conv1(x)
        return x
