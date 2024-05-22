from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')
img = Image.open('images/cheess.png')

# ToTensor的使用
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img)
writer.add_image('img_tensor', img_tensor)

# Normalize的使用
print(img_tensor[0][0][0])
tensor_norm = transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
img_norm = tensor_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('img_norm', img_norm)

# Resize的使用
# print(img.size)
tensor_resize = transforms.Resize((250, 400))
img_resize = tensor_resize(img_tensor)
writer.add_image('img_resize', img_resize, 1)

# Compose的使用
# 将 tensor_resize_2 和 tensor_trans 两个操作组合，注意顺序
tensor_resize_2 = transforms.Resize((512))
tensor_compose = transforms.Compose([tensor_resize_2, tensor_trans])
img_resize_2 = tensor_compose(img)
writer.add_image('img_resize_2', img_resize_2)

# RandomCrop的使用
tensor_random_crop = transforms.RandomCrop((100, 100))
tensor_compose_2 = transforms.Compose([tensor_random_crop, tensor_trans])
# img_random_crop = tensor_random_crop(img)
# print(type(img_random_crop))
for i in range(10):
    img_random_crop = tensor_compose_2(img)
    writer.add_image('img_random_crop', img_random_crop, i)

writer.close()