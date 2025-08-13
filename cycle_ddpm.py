import torch
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import nn, Tensor
from config import Configs
from diffusion_unet import DFUNet
from plot import disply_images
from scheduler import DDPMScheduler




class CycleDDPM(nn.Module):
    def __init__(self, config: Configs) -> None:
        super().__init__()
        self.config = config
        self.scheduler = DDPMScheduler(config)
        self.model = DFUNet(
            ch_input=config.output_channel,  # MRI通道数
            ch_output=config.output_channel,
            condition_channels=config.input_channel,  # CT作为条件
            image_size=config.image_size,
            layers=config.layers,
        )
        # 添加循环一致性的反向模型
        self.reverse_model = DFUNet(
            ch_input=config.input_channel,  # 反向输入通道数
            ch_output=config.input_channel,  # 反向输出通道数
            condition_channels=config.output_channel,  # MRI作为条件
            image_size=config.image_size,
            layers=config.layers,
        )

    def forward_diffusion_to_mri(self, x: Tensor, ts: Tensor) -> Tuple[Tensor, Tensor]:
        """生成mri正向过程"""
        noise = torch.randn_like(x)
        x_noisy = self.scheduler.add_noise(x, noise, ts)
        return x_noisy, noise

    def forward_diffusion_to_ct(self, x: Tensor, ts: Tensor) -> Tuple[Tensor, Tensor]:
        """生成ct正向过程"""
        noise = torch.randn_like(x)
        x_noisy = self.scheduler.add_noise(x, noise, ts)
        return x_noisy, noise

    def predict_noise_ct(self, x: Tensor, ts: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        """使用reverse_model预测CT噪声(MRI作为条件)"""
        return self.model(x, ts, condition=condition)

    def predict_noise_mri(self, x: Tensor, ts: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        """使用model预测MRI噪声(CT作为条件)"""
        return self.reverse_model(x, ts, condition=condition)

    def generate_ct(self, mri_image: Tensor) -> Tensor:
        """生成CT图像 (输入:MRI图像,输出:CT图像)"""
        x = torch.rand_like(mri_image).to(device=mri_image.device)  # 初始随机噪声

        # 获取时间步序列（从T到1）
        timesteps = self.scheduler.set_timesteps()

        # 逐步去噪
        for t in timesteps:
            t_batch = torch.full((mri_image.shape[0],), t.item(), device=mri_image.device, dtype=torch.long)

            with torch.no_grad():
                predicted_noise = self.predict_noise_ct(x, t_batch, condition=mri_image)

            x = self.scheduler.step(predicted_noise, t_batch, x)

        return x

    def generate_mri(self, ct_image: Tensor) -> Tensor:
        """生成MRI图像(输入:CT图像,输出:MRI图像)"""
        x = torch.rand_like(ct_image).to(device=ct_image.device)  # 初始随机噪声

        # 获取时间步序列（从T到1）
        timesteps = self.scheduler.set_timesteps()

        # 逐步去噪
        for t in timesteps:
            t_batch = torch.full((ct_image.shape[0],), t.item(), device=ct_image.device, dtype=torch.long)

            with torch.no_grad():
                predicted_noise = self.predict_noise_mri(x, t_batch, condition=ct_image)

            x = self.scheduler.step(predicted_noise, t_batch, x)

        return x
    def generate_ct_partial_grad(self, mri_image: Tensor, K:int=1):
        """循环一致性损失计算时，只误差逆传播最后K步"""
        x = torch.rand_like(mri_image)
        timesteps = self.scheduler.set_timesteps()
        for i, t in enumerate(timesteps):
            t_batch = torch.full((mri_image.size(0),), t.item(), device=mri_image.device, dtype=torch.long)
            if i < len(timesteps) - K:
                with torch.no_grad():
                    predicted_noise = self.predict_noise_ct(x, t_batch, condition=mri_image)
            else:
                predicted_noise = self.predict_noise_ct(x, t_batch, condition=mri_image)
            x = self.scheduler.step(predicted_noise, t_batch, x)
        return x

    def generate_mri_partial_grad(self, ct_image: Tensor, K:int=10):
        """循环一致性损失计算时，只误差逆传播最后K步"""
        x = torch.rand_like(ct_image)
        timesteps = self.scheduler.set_timesteps()
        for i, t in enumerate(timesteps):
            t_batch = torch.full((ct_image.size(0),), t.item(), device=ct_image.device, dtype=torch.long)
            if i < len(timesteps) - K:
                with torch.no_grad():
                    predicted_noise = self.predict_noise_mri(x, t_batch, condition=ct_image)
            else:
                predicted_noise = self.predict_noise_mri(x, t_batch, condition=ct_image)
            x = self.scheduler.step(predicted_noise, t_batch, x)
        return x

    def generate_ct_with_grad(self, mri_image):
        """
        使用 梯度检查点（torch.utils.checkpoint）在每个时间步节省显存，只在需要的时候重新计算前向。配合 混合精度训练（amp.autocast）减少显存开销。
        """
        x = torch.rand_like(mri_image).requires_grad_(True)
        timesteps = self.scheduler.set_timesteps()
        for t in timesteps:
            predicted_noise = cp.checkpoint(self.predict_noise_ct, x, t[None].to(device=mri_image.device), mri_image,use_reentrant=False)
            x = self.scheduler.step(predicted_noise, t, x)
        return x
    def generate_mri_with_grad(self, ct_image):
        x = torch.rand_like(ct_image).requires_grad_(True)
        timesteps = self.scheduler.inf_timesteps
        for t in timesteps:
            predicted_noise = cp.checkpoint(self.predict_noise_mri, x, t[None].to(device=ct_image.device), ct_image,use_reentrant=False)
            x = self.scheduler.step(predicted_noise, t, x)

        return x

    def gray_scale_consistency_loss(self, generated_img, target_img):
        """
        计算灰度分布一致性损失
        """
        # 计算图像的均值和标准差
        gen_mean = torch.mean(generated_img, dim=[2, 3], keepdim=True)
        gen_std = torch.std(generated_img, dim=[2, 3], keepdim=True)

        target_mean = torch.mean(target_img, dim=[2, 3], keepdim=True)
        target_std = torch.std(target_img, dim=[2, 3], keepdim=True)

        # 分布一致性损失
        mean_loss = F.mse_loss(gen_mean, target_mean)
        std_loss = F.mse_loss(gen_std, target_std)

        return mean_loss + std_loss
    def cycle_consistency_loss(self, x_A: Tensor, x_B: Tensor,lambda_gray: float = 1) -> Tensor:
        """计算循环一致性损失"""
        # with torch.no_grad():
        # ct -> mri -> ct
        x_B_pred = self.generate_mri_with_grad(x_A)
        x_A_reconstructed = self.generate_ct_with_grad(x_B_pred)

        # mri -> ct -> mri
        x_A_pred = self.generate_ct_with_grad(x_B)
        x_B_reconstructed = self.generate_mri_with_grad(x_A_pred)
        with torch.no_grad():
            images = torch.cat([x_A, x_B_pred, x_A_reconstructed, x_B, x_A_pred, x_B_reconstructed], dim=0)
            disply_images(images, row_num=3, title="cycle_loss_generate_display",save_dir='D:\\0-nebula\\dataset\\results')
        # 计算一致性损失
        cycle_loss_A = F.l1_loss(x_A, x_A_reconstructed)
        cycle_loss_B = F.l1_loss(x_B, x_B_reconstructed)
        gray_loss_A = self.gray_scale_consistency_loss(x_B_pred, x_B)  # MRI生成的灰度一致性
        gray_loss_B = self.gray_scale_consistency_loss(x_A_pred, x_A)  # CT生成的灰度一致性
        # 计算灰度损失
        return cycle_loss_A + cycle_loss_B + lambda_gray*(gray_loss_A + gray_loss_B)
        # return cycle_loss_A
    def compute_loss(self, x_A: Tensor, x_B: Tensor, lambda_cycle: float = 1,epoch:int = 200,epochs:int=400) -> dict[str, Tensor]:
        """计算损失,其中输入的x_A是ct图像，x_B是mri图像"""
        batch_size = x_A.shape[0]
        timesteps = self.scheduler.sample_timesteps(batch_size)

        # debug
        # images = torch.cat([x_A, x_B], dim=0)
        # disply_images(images,title="原始CT和MRI图像")

        # 生成mri图像,生成ct图像的添加噪声操作
        x_noisy_mri, noise_mri = self.forward_diffusion_to_mri(x_A, timesteps)
        predicted_noise_mri = self.predict_noise_mri(x_noisy_mri, timesteps, condition=x_B)  # 输入的x_A为ct，作为训练条件

        # 生成ct图像,生成mri图像的添加噪声操作
        x_noisy_ct, noise_ct = self.forward_diffusion_to_ct(x_B, timesteps)
        predicted_noise_ct = self.predict_noise_ct(x_noisy_ct, timesteps, condition=x_A)  # 输入的x_B为mri，作为训练条件

        # debug
        # all_images = torch.cat([x_A[0:1],x_B[0:1],x_noisy_ct[0:1],x_noisy_mri[0:1]], dim=0)
        # disply_images(all_images,title="生成图像")

        loss_A = F.mse_loss(noise_mri, predicted_noise_mri)
        loss_B = F.mse_loss(noise_ct, predicted_noise_ct)

        # 计算循环一致性损失
        if epoch >= epochs * 3 // 4 :
            cycle_loss = self.cycle_consistency_loss(x_A, x_B)
            total_loss = loss_A + loss_B + lambda_cycle * cycle_loss
        else:
            cycle_loss = torch.Tensor([0])
            total_loss = loss_A + loss_B
        return {"loss": total_loss, "loss_A": loss_A, "loss_B": loss_B, "cycle_loss": cycle_loss}