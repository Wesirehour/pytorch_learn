import PIL.Image
import torch
import torchvision.transforms

imgs_path = './imgs/plane.jpg'
img = PIL.Image.open(imgs_path)
img = img.convert('RGB')  # png图片有第四个通道：透明度，改成三通道
tran = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
img = tran(img)
model = torch.load('model_4.pth').to('cuda')
# print(model)
img = torch.reshape(img, (1,3,32,32)).to('cuda')
model.eval()
with torch.no_grad():
    output = model(img)
# print(output)
print('飞机' if output.argmax(1).item() == 0 else '其他')