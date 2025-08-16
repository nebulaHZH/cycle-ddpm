# 数据集加载
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image,ImageReadMode
from torchvision.transforms import v2,InterpolationMode
from PIL import Image
from torchvision import transforms
from typing import Tuple
import matplotlib.pyplot as plt

class PairedImageDataset(Dataset):
    def __init__(self, dir_A, dir_B, image_size=256, gray_scale=True):
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.image_size = image_size
        self.gray_scale = gray_scale
        
        self.transform = v2.Compose([
            v2.Resize((self.image_size, self.image_size),InterpolationMode.BILINEAR),
            v2.ToTensor(),
            v2.Normalize([0.5], [0.5])  # 映射到 [-1, 1] 范围
        ])
        # 获取并排序文件名，确保配对
        files_A = sorted([f for f in os.listdir(dir_A) if f.endswith('.png')])
        files_B = sorted([f for f in os.listdir(dir_B) if f.endswith('.png')])

        # 确保两个目录有相同的文件
        assert len(files_A) == len(files_B), "目录A和B的文件数量不匹配"
        assert files_A == files_B, "文件名不匹配"
        
        self.filenames = files_A
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx) ->Tuple[torch.Tensor,torch.Tensor]:
        filename = self.filenames[idx]
        
        # 加载配对图像
        img_A = Image.open(os.path.join(self.dir_A, filename)).convert('L')
        img_B = Image.open(os.path.join(self.dir_B, filename)).convert('L')

        image_A:torch.Tensor = self.transform(img_A)
        image_B:torch.Tensor = self.transform(img_B)

        return image_A, image_B
