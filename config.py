from typing import NamedTuple

import torch

class Configs:
    def __init__(self,
                 dir_a_path: str = "D:/0-nebula/dataset/ixi_paried/t1_30_resized",
                 dir_b_path: str = "D:/0-nebula/dataset/ixi_paried/t2_30_resized",
                 lr: float = 2e-5,
                 batch: int = 2,
                 epochs: int = 100,
                 clip: float = 1.0,
                 num_train_timesteps: int =400,
                 num_inference_timesteps: int = 200,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.005,
                 image_size: int = 256,
                 ch_input: int = 1,
                 device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 current_epoch: int = 0,
                 input_channel: int = 1,
                 output_channel: int = 1,
                 layers: int = 2,
                 time_step: int = 1000
        ):
        self.dir_a_path = dir_a_path # CT数据集路径
        self.dir_b_path = dir_b_path # MR数据集路径
        self.lr = lr # 学习率
        self.batch = batch # 批量大小
        self.epochs = epochs # 训练轮数
        self.clip = clip # 梯度裁剪
        self.num_train_timesteps = num_train_timesteps # 训练时间步
        self.num_inference_timesteps = num_inference_timesteps # 推理时间步
        self.beta_start = beta_start # 时间步开始
        self.beta_end = beta_end # 时间步结束
        self.image_size = image_size # 图像大小
        self.ch_input = ch_input # 输入通道数
        self.device = device # 设备
        self.current_epoch = current_epoch # 当前轮数
        self.input_channel = input_channel # 输入通道数
        self.output_channel = output_channel # 输出通道数
        self.layers = layers # 层数
        self.time_step = time_step # 时间步
