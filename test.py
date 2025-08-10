from matplotlib import pyplot as plt
import numpy as np
import torch
from cycle_ddpm import CycleDDPM
from PIL import Image

from plot import Plotter
from scheduler import DDPMScheduler

def ct_to_mri_generation(model:CycleDDPM, ct_image_path, scheduler:DDPMScheduler,original_image_path:str):
    """使用训练好的CycleDDPM模型将CT图像转换为MRI图像"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载并预处理CT图像
    ct_image = Image.open(ct_image_path).convert('L')  # 确保是灰度图
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
    # for t in scheduler.inf_timesteps:
    #     # 使用模型的generate_A_to_B方法进行转换
    #     with torch.no_grad():
    #         noise_pred = model.generate_B_to_A(ct_tensor)
    #         noise_sample = scheduler.step(noise_pred,t, ct_tensor)
    #     if t % 400 == 0: 
    #         images = torch.concat([images, noise_sample],dim=0)
    images = torch.concat([images, model.generate_A_to_B(images)],dim=0)
    images = torch.concat([images, origin_image],dim=0)
    # 后处理：转换回[0, 255]范围
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    images = (images * 255).type(torch.uint8)
    Plotter(images,f"timestep={len(scheduler.inf_timesteps)},per 1000").plot()
    return images

import torch
from cycle_ddpm import CycleConfig, CycleDDPM

# 1. 创建与训练时相同的配置
config = CycleConfig(
    data_path="D:\\0-nebula\\dataset\\Havard\\MyDatasets\\CT-MRI\\train\\MRI",
    image_size=256,
    num_classes=2,
    batch=1,
    epochs=200,
    lr=1e-4,
    save_period=10,
    proj_name="test",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    sample_period=10,
    clip=1.0,
    num_train_timesteps=2000,
    num_inference_timesteps=20,
    beta_start=0.0001,
    beta_end=0.005,
)

# 2. 创建模型实例
model = CycleDDPM(config).to(config.device)
scheduler = DDPMScheduler(config)
# 3. 加载保存的模型状态
checkpoint_path = r"checkpoints/20"  # 假设你想加载第100轮的模型
checkpoint = torch.load(checkpoint_path, map_location=config.device)

# 4. 恢复模型状态
model.load_state_dict(checkpoint['model_state_dict'])

# 5. 如果需要继续训练，还需要创建优化器并恢复其状态
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 6. 获取保存时的损失值（如果需要）
loss = checkpoint['loss']

# 现在模型已经加载完成，可以用于推理或继续训练
model.eval()  # 如果用于推理
# 或者 model.train() # 如果继续训练

ct_image_path = 'D:/0-nebula/dataset/ixi_paried/test_t2/0202.png'
original_image_path = 'D:/0-nebula/dataset/ixi_paried/test_t1/0202.png'
generated_mri = ct_to_mri_generation(model, ct_image_path, scheduler=scheduler,original_image_path=original_image_path)

# generated_mri.save('generated_mri.png')


ct_image = Image.open(ct_image_path)

# 将原始图像和生成的图像放在一起展示
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(ct_image, cmap='gray')
# plt.title('CT Image')
# plt.subplot(1, 2, 2)
# plt.imshow(generated_mri, cmap='gray')
# plt.title('Generated MRI')
# plt.show()