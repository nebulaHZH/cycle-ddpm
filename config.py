from typing import NamedTuple

import torch

class Configs(NamedTuple):
    """配置类基类"""
    proj_name: str = "test"
    lr: float = 2e-5  # 降低学习率
    batch: int = 2  # 减小批量大小
    epochs: int = 100  # 增加训练轮数
    save_period: int = 10
    sample_period: int = 10
    clip: float = 1.0
    num_train_timesteps: int = 2000
    num_inference_timesteps: int = 200
    beta_start: float = 0.0001
    beta_end: float = 0.005
    data_path: str = "D:\\0-nebula\\dataset\\Havard\\MyDatasets\\CT-MRI\\train\\MRI"
    image_size: int = 256
    num_classes: int = 2
    ch_input: int = 1
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    current_epoch: int = 0  # 添加当前epoch记录
