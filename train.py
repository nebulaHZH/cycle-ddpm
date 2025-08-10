import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from load import ImageData
from torch.optim.lr_scheduler import CosineAnnealingLR
from scheduler import DDPMScheduler
from add_nosie import generate_image
from torch import nn
from diffusion_unet import DFUNet
from config import Configs
config = Configs(
    data_path="D:\\0-nebula\\dataset\\Havard\\MyDatasets\\CT-MRI\\train\\MRI",
    image_size=256,
    num_classes=2,
    batch=4,
    epochs=200,
    lr=1e-4,
    save_period=10,
    proj_name="test",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    sample_period=10,
    clip=1.0,
    num_train_timesteps=2000,
    num_inference_timesteps=200,
    beta_start=0.0001,
    beta_end=0.005,
)


dataset = ImageData(
    image_dir='D:\\0-nebula\\dataset\\Havard\\MyDatasets\\CT-MRI\\test\\CT',
    image_size=256,
    gray_scale=True,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    load_all=False
)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = DFUNet(
    ch_input=config.ch_input,
    ch_output=1,
    image_size=config.image_size,
).to(config.device)
scheduler = DDPMScheduler(config)

# diffusers 里面用的是 AdamW
# lr 不能设置的太大
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
scheduler_lr = CosineAnnealingLR(optimizer, T_max=config.epochs)
training_data = ImageData(config.data_path, config.image_size)
train_dataloader = DataLoader(training_data, batch_size=config.batch, shuffle=True)
# 显示原图像和加噪后的图像
l = [training_data[i].unsqueeze(0).to(config.device) for i in range(1)]  # 1，1，256，256
test_images = torch.concat(tensors=l,dim=0) # 4,1,256,256
test_labels = torch.concat(tensors=l,dim=0)
timesteps = scheduler.sample_timesteps(10)
noise = torch.randn(test_images.shape).to(config.device)
noisy_image = scheduler.add_noise(image=test_images, noise=noise, timesteps=timesteps)
# Plotter(noisy_image, "noisy image", config.proj_name).plot()
# Plotter(test_images, "original image", config.proj_name).plot()   
#plot_images((test_images / 2 + 0.5).clamp(0, 1), titles=test_labels.detach().tolist(), fig_titles="original image", save_dir=config.proj_name)
#plot_images((noisy_image / 2 + 0.5).clamp(0, 1), titles=test_labels.detach().tolist(), fig_titles="noisy image", save_dir=config.proj_name)

# 训练模型
# for ep in range(config.epochs):
#     progress_bar = tqdm(train_dataloader)
#     model.train()
#     avg_loss = 0
#     count = 0
#     for image in progress_bar:
#         batch = image.shape[0]
#         image = image.to(config.device)
#         timesteps = scheduler.sample_timesteps(batch).to(device=config.device)
#         noise = torch.randn(image.shape).to(config.device)
#         noisy_image = scheduler.add_noise(image=image, noise=noise, timesteps=timesteps)

#         pred = model(noisy_image, timesteps)[0]
#         loss = torch.nn.functional.mse_loss(pred, noise)
#         optimizer.zero_grad()
#         loss.backward()

#         # gradient clipping, 用来防止 exploding gradients
#         nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         avg_loss = avg_loss + loss.detach().item()
#         count = count + 1
#         logs = {"avg_loss": avg_loss / count, "ep": ep+1,"lr": optimizer.param_groups[0]['lr']}
#         progress_bar.set_postfix(**logs)
#     scheduler_lr.step()
# # 保存模型
# torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#     }, r"checkpoints/" + str(ep+1))

# checkpoint  = torch.load("checkpoints/100")
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
# # 根据保存的模型生成数据集图像
# # 模型生成tensor
# images = generate_image(
#     model=model,
#     scheduler=scheduler
#     )
# Plotter(images,f"epoch_{config.epochs+1}").plot()