import train_ddpm
import torchvision.transforms as transforms
import torch
from cycle_ddpm import CycleDDPM
import torch.nn.functional as F
from PIL import Image
from plot import  disply_images
from scheduler import DDPMScheduler


def ct_to_mri_generation(model:CycleDDPM, ct_image_path:str, scheduler:DDPMScheduler,original_image_path:str):
    """使用训练好的CycleDDPM模型将CT图像转换为MRI图像"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 使用与训练时相同的变换
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 与训练时保持一致
    ])
    # 加载并预处理CT图像
    ct_image = Image.open(ct_image_path).convert('L')  # 确保是灰度图
    ct_tensor = transform(ct_image).unsqueeze(0).to(device)
    origin_image = Image.open(original_image_path).convert('L')
    origin_tensor = transform(origin_image).unsqueeze(0).to(device)
    images = ct_tensor
    origin_image = origin_tensor
    res = model.generate_mri_with_grad(images, is_inference=True)
    images = torch.concat([images, res], dim=0)
    images = torch.concat([images, origin_image], dim=0)
    # 后处理：反归一化回 [0, 1] 范围用于显示
    # 对于 Normalize([0.1883], [0.2469])，反归一化公式为: x = (normalized_x * 0.2469) + 0.1883
    images = images * 0.5 + 0.5
    images = torch.clamp(images, 0, 1)
    disply_images(images=images,row_num=3, title="origin(left),generate(mid),label(right)")
    return images



# 1. 创建与训练时相同的配置
config = train_ddpm.config


model = CycleDDPM(config).to(config.device)
scheduler = DDPMScheduler(config)

checkpoint_path = r"D:\\0-nebula\\dataset\\checkpoints\\40"  # 假设你想加载第100轮的模型
checkpoint = torch.load(checkpoint_path, map_location=config.device)

model.load_state_dict(checkpoint['model_state_dict'])

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

loss = checkpoint['loss']
model.eval()
ct_image_path = 'D:/0-nebula/dataset/ixi_paried/test_t1/0201.png'
original_image_path = 'D:/0-nebula/dataset/ixi_paried/test_t2/0201.png'
generated_mri = ct_to_mri_generation(model, ct_image_path, scheduler=scheduler,original_image_path=original_image_path)

# generated_mri.save('generated_mri.png')


ct_image = Image.open(ct_image_path)
