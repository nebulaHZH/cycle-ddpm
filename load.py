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
class ImageData(Dataset):
    def __init__(self,
                image_dir:str,
                image_size:int,
                gray_scale:bool = False,
                device:torch.device = torch.device('cpu'),
                load_all:bool = False,)->None:
        self.image_paths = [f.path for f in os.scandir(image_dir) if os.path.isfile(f) and os.path.splitext(f)[-1].lower() in [".jpg", ".png", ".jpeg", ".bmp"]]
        self.target_size = image_size
        self.transforms = v2.Compose([
            v2.Resize((self.target_size, self.target_size),InterpolationMode.BILINEAR),
            v2.PILToTensor(),
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize([127.5],[127.5])
        ])
        self.device = device
        self.load_all = load_all
        if load_all:
            self.images = [self.transforms(read_image(path,mode=ImageReadMode.GRAY if not gray_scale else ImageReadMode.GRAY).float()) for path in self.image_paths]
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index:int)->torch.Tensor:
        if self.load_all:
            return self.images[index]
        else:
            image = read_image(self.image_paths[index],mode=ImageReadMode.GRAY).float()
            image:torch.Tensor = self.transforms(image)
            return image.to(self.device)

class PairedImageDataset(Dataset):
    def __init__(self, dir_A, dir_B, image_size=256, gray_scale=True):
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.image_size = image_size
        self.gray_scale = gray_scale
        
        self.transform = v2.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 归一化到[-1,1]
        ])
        # 获取并排序文件名，确保配对
        files_A = sorted([f for f in os.listdir(dir_A) if f.endswith('.png')])
        files_B = sorted([f for f in os.listdir(dir_B) if f.endswith('.png')])
        # for i in range(len(files_A)):
        #     if files_A[i] != files_B[i]:
        #         raise ValueError(f"文件名不匹配：{files_A[i]} != {files_B[i]}")
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


        # 预处理
        img_A = img_A.resize((self.image_size, self.image_size))
        img_B = img_B.resize((self.image_size, self.image_size))
        

        image_A:torch.Tensor = self.transform(img_A)
        image_B:torch.Tensor = self.transform(img_B)

        return image_A, image_B




# # 测试使用
# if __name__ == '__main__':
#     dataset = ImageData(
#         image_dir='D:\\0-nebula\\dataset\\Havard\\MyDatasets\\CT-MRI\\test\\CT',
#         image_size=256,
#         gray_scale=True,
#         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#         load_all=False
#     )
#     data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
#     # 显示一个batch的形状
#     image:torch.Tensor = next(iter(data_loader)) 
#     image = image[0]
#     # 展示图片
#     import matplotlib.pyplot as plt
#     plt.imshow(image.cpu().permute(1,2,0).numpy(), cmap='gray')
#     plt.show()
