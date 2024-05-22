from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter('logs')
img_PIL = Image.open('data/train/bees_image/16838648_415acd9e3f.jpg')
img_array = np.array(img_PIL)
writer.add_image('train', img_array, 2, dataformats='HWC')
for i in range(100):
    writer.add_scalar('y = 2x', 3*i, i)

writer.close()