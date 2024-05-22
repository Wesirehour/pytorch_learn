from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_path = 'data/train/ants_image/0013035.jpg'
img = Image.open(img_path)
# ToTensor使用，将PIL image和numpy array转换为tensor

# 创建工具
tensor_trans = transforms.ToTensor()
# 使用工具
tensor_img = tensor_trans(img)

writer = SummaryWriter('logs')
writer.add_image('tensor_img', tensor_img)
writer.close()
