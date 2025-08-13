import torch
from config import Configs
import numpy

class Scheduler:
    """基础的diffusion model 过程"""
    def __init__(self, config:Configs) -> None:
        self.config = config

        # 设置训练和推理的步数
        self.num_train_timesteps: int = self.config.num_train_timesteps
        self.num_inference_steps: int = self.config.num_inference_timesteps
        self.set_timesteps()        # 初始化 timesteps

        # 创建从 start 到 end 的一个 beta 的数组
        self.beta_start: float = self.config.beta_start
        self.beta_end: float = self.config.beta_end
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps, dtype=torch.float32).to(self.config.device)
        self.alphas = 1.0 - self.betas      # alpha = 1 - beta
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)     # alphas_cumprod 即 α_bar

    def set_timesteps(self):
        '''
        生成一个时间序列的数组
        '''
        step_ratio = self.num_train_timesteps // self.num_inference_steps # 时间步长
        timesteps = (numpy.arange(0, self.num_inference_steps) * step_ratio).round()[::-1].copy().astype(numpy.int64)
        self.inf_timesteps = torch.from_numpy(timesteps).to(self.config.device)
        return self.inf_timesteps

    def add_noise(self,image: torch.Tensor,noise: torch.Tensor,timesteps: torch.Tensor) -> torch.Tensor:
        '''
        根据采样的时间点，对图像进行加噪
        '''
        # √ᾱ_t
        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod[timesteps])      
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()  
        '''将系数扩展到与图像的维度一致

            初始:sqrt_alpha_prod.shape = [batch]
            扩展后:sqrt_alpha_prod.shape = [batch, 1, 1, 1]
        '''
        while len(sqrt_alpha_prod.shape) < len(image.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        # √(1-ᾱ_t)
        sqrt_one_minus_alpha_prod = torch.sqrt((1 - self.alphas_cumprod[timesteps]))     
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(image.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        # √ᾱ_t*x_0 + √(1-ᾱ_t)*ε
        noisy_samples = sqrt_alpha_prod * image + sqrt_one_minus_alpha_prod * noise 
        return noisy_samples

    def sample_timesteps(self, size: int):
        ''' 
        采样一些随机的时间点
        '''
        timesteps = torch.randint(0, self.num_train_timesteps, (size,), device=self.config.device).long()
        timesteps = timesteps.to(device=self.config.device)
        return timesteps

    def prev_timestep(self, timestep: torch.Tensor):
        ''' 
        获取 timestep t 的前一个时间点，即 t-Δt
        Δt = num_train / num_inf
        '''
        return timestep - self.num_train_timesteps // self.num_inference_steps

    def step(self, noise_pred: torch.Tensor, timestep: torch.Tensor, noisy_image: torch.Tensor):
        ...

class DDPMScheduler(Scheduler):
    def __init__(self, config:Configs) -> None:
        super().__init__(config)

    def step(self, noise_pred: torch.Tensor, timestep: torch.Tensor, noisy_image: torch.Tensor):
        # 计算前一时刻的时间步
        prev_t = self.prev_timestep(timestep)

        alpha_bar_at_t = self.alphas_cumprod[timestep] #ᾱt
        # alpha_bar_at_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0) 
         # 对每个时间步单独处理边界情况
        alpha_bar_at_prev_t = torch.where(prev_t >= 0, 
                                      self.alphas_cumprod[prev_t], 
                                      torch.tensor(1.0, device=self.config.device))#ᾱ(t-1)
        beta_bar_at_t = 1 - alpha_bar_at_t # 1-ᾱt
        beta_bar_at_prev_t = 1 - alpha_bar_at_prev_t # 1-ᾱ(t-1)

        current_alpha_t = alpha_bar_at_t / alpha_bar_at_prev_t # ᾱt/ᾱ(t-1)
        current_beta_t = 1 - current_alpha_t # 1-ᾱt/ᾱ(t-1)

        # 根据噪声预测 x0 , 去噪后的图像为denoised_image
        # x_0 = ( x_t - √1-α_bar_t ε ) / √α_bar_t
        # x_t = √α_bar_t x_0 + √1-α_bar_t ε
        denoised_image = (noisy_image - torch.sqrt(beta_bar_at_t) * noise_pred) / torch.sqrt(alpha_bar_at_t)

        # 将图像范围限制在 [-1, 1]
        denoised_image = denoised_image.clamp(-self.config.clip, self.config.clip)

        # 根据公式计算均值 ~μ，
        # 这里也可以根据 μ=1/√α_t (x_t - (1-α_t)/√1-α_bar_t ε_t) 得到
        pred_original_sample_coeff = (torch.sqrt(alpha_bar_at_prev_t) * current_beta_t) / beta_bar_at_t
        current_sample_coeff = torch.sqrt(current_alpha_t) * beta_bar_at_prev_t / beta_bar_at_t
        pred_prev_image = pred_original_sample_coeff * denoised_image + current_sample_coeff * noisy_image

        # 加入噪声 σ_t z
        # 其中, σ_t^2 = ( 1-α_bar_(t-1)/1-α_bar_t ) β_t
        variance = 0
        if timestep > 0:
            z = torch.randn(noise_pred.shape).to(device=self.config.device)
            variance = (1 - alpha_bar_at_prev_t) / (1 - alpha_bar_at_t) * current_beta_t
            variance = torch.clamp(variance, min=1e-20)
            variance = torch.sqrt(variance) * z

        return pred_prev_image + variance