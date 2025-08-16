# 画图函数

from matplotlib import pyplot, pyplot as plt
from typing import List, Optional
import torchvision.utils as vutils
import numpy
import math
import torch
import os


def disply_images(images: torch.Tensor , save_dir: Optional[str] = None ,row_num: int = 2, title: Optional[str] = None)->None:
    """画出图像，images的维度是[num,c,w,h],图片在第一维拼接"""
    grid = vutils.make_grid(images, nrow=row_num, normalize=True, scale_each=True)
    plt.figure(figsize=(10, 5))
    plt.title(title, fontsize=10)
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.title("Real CT (Left) and Real MRI (Right)")
    plt.axis("off")
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
        plt.savefig(os.path.join(save_dir, title + ".png"))
    else:
        plt.show()
    plt.close()  # 添加这一行来关闭图形
def plot_loss(loss_plot:dict[str,list]):
    # 根据loss_plot列表画出损失曲线图
    epochs = list(range(1, len(loss_plot['loss']) + 1))
    total_losses = loss_plot["loss"]
    a_losses = loss_plot["loss_A"]
    b_losses = loss_plot["loss_B"]
    cycle_losses = loss_plot["cycle_loss"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training Loss Curves", fontsize=16)
    # 绘制总损失
    axes[0, 0].plot(epochs, total_losses, 'b-', linewidth=2)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # 绘制Loss A
    axes[1, 0].plot(epochs, a_losses, 'r-', linewidth=2)
    axes[1, 0].set_title('Loss A')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    # 绘制Loss B
    axes[1, 1].plot(epochs, b_losses, 'g-', linewidth=2)
    axes[1, 1].set_title('Loss B')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)

    # 绘制Cycle Loss
    axes[0, 1].plot(epochs, cycle_losses, 'm-', linewidth=2)
    axes[0, 1].set_title('Cycle Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('examples/training_loss_subplots.png', dpi=300, bbox_inches='tight')
    plt.show()