import numpy as np
import train_ddpm
import torch.nn.functional as F
from cycle_ddpm import CycleDDPM
from PIL import Image
from plot import  disply_images
from scheduler import DDPMScheduler


def ct_to_mri_generation(model:CycleDDPM, ct_image_path:str, scheduler:DDPMScheduler,original_image_path:str):
    """使用训练好的CycleDDPM模型将CT图像转换为MRI图像"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载并预处理CT图像
    ct_image = Image.open(ct_image_path).convert('L') # 确保是灰度图
    ct_image = ct_image.resize((256, 256))  # 调整为模型要求的尺寸
    ct_array = np.array(ct_image)

    origin_image = Image.open(original_image_path).convert('L')
    origin_image = origin_image.resize((256, 256))
    origin_array = np.array(origin_image)

    # 转换为tensor并添加批次和通道维度
    ct_tensor = torch.from_numpy(ct_array).unsqueeze(0).unsqueeze(0).float()
    ct_tensor = ct_tensor.to(device)

    origin_tensor = torch.from_numpy(origin_array).unsqueeze(0).unsqueeze(0).float()
    origin_tensor = origin_tensor.to(device)

    # 归一化到[-1, 1]范围（与训练时一致）
    ct_tensor = (ct_tensor / 127.5) - 1
    origin_tensor = (origin_tensor / 127.5) - 1
    images =  ct_tensor
    origin_image = origin_tensor
    res = model.generate_ct_with_grad(images,is_inference=True)
    images = torch.concat([images, res],dim=0)
    images = torch.concat([images, origin_image],dim=0)
    # 后处理：转换回[0, 255]范围
    images = (images + 1) / 2   
    images = torch.clamp(images, 0, 1)
    disply_images(images=images,row_num=3, title="origin(left),generate(mid),label(right)")
    return images

import torch
from cycle_ddpm import CycleDDPM

# 1. 创建与训练时相同的配置
config = train_ddpm.config


model = CycleDDPM(config).to(config.device)
scheduler = DDPMScheduler(config)

checkpoint_path = r"E:\\nebula\\checkpoints\\100"  # 假设你想加载第100轮的模型
checkpoint = torch.load(checkpoint_path, map_location=config.device)

model.load_state_dict(checkpoint['model_state_dict'])

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

loss = checkpoint['loss']
model.eval()
ct_image_path = 'E:/nebula/test_t1/0201.png'
original_image_path = 'E:/nebula/test_t2/0201.png'
generated_mri = ct_to_mri_generation(model, ct_image_path, scheduler=scheduler,original_image_path=original_image_path)

# generated_mri.save('generated_mri.png')


ct_image = Image.open(ct_image_path)
