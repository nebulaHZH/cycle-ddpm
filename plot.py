# 画图函数

from matplotlib import pyplot, pyplot as plt
from typing import List, Optional
import torchvision.utils as vutils
import numpy
import math
import torch
import os

class Plotter:
    def __init__(self, images:torch.Tensor, title: str,save_dir: Optional[str | None] = None) -> None:
        self.images = images
        self.title = title
        self.save_dir = save_dir
    
    def plot(self):
        d = self.images.dim()
        print(self.images.shape)
        if d == 4:
            i = self.images.shape[0]
            images = self.images.split(1,dim=0)
        else:
            images = [self.images]
        if len(images) == 1:
            rows = 1
            cols = 1
        else:
            rows = (len(images)+1) // 2
            cols = 2
        fig, axes = pyplot.subplots(rows, cols, 
                                figsize=(cols * 4, rows * 4),  # 每个子图4英寸
                                squeeze=False)
        for i, image in enumerate(images):
            # print(image.squeeze(0).shape)
            img = image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            axes[i // cols, i % cols].imshow(img,cmap='gray')
            axes[i // cols, i % cols].set_title(f"image {i},length:{len(images)}")
            axes[i // cols, i % cols].axis('off')
        fig.suptitle(self.title)
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)  # 确保目录存在
            pyplot.savefig(os.path.join(self.save_dir, self.title + ".png"))
        pyplot.show()


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
def plot_images(
        images: 'torch.Tensor',
        titles: Optional[List[str]] = None,
        fig_titles: Optional[str] = None,
        save_title: Optional[str] = None,
        save_dir: Optional[str] = None,
        cols: int = 4):

    _images = images
    b, c, h, w = _images.shape

    # 当只有一个通道时，将一个通道复制为3个
    if c == 1:
        _images = torch.repeat_interleave(_images, 3, dim=1)
    if c > 3:
        _images = _images[:, :3, :, :]

    # 计算行列数
    COLS = cols
    ROWS = int(math.ceil(b / COLS))

    # 将图像数据转为 numpy 并调整维度
    if torch.is_tensor(_images):
        image_array = _images.detach().cpu().numpy()
    else:
        image_array = _images  # 防止输入不是tensor时出错

    _image_array = numpy.transpose(image_array, [0, 2, 3, 1])

    fig, axes = pyplot.subplots(ROWS, COLS, figsize=(COLS, ROWS))
    pyplot.subplots_adjust(wspace=0.05, hspace=0.05)

    if fig_titles is not None:
        fig.suptitle(fig_titles, fontsize=10)

    if titles is None:
        titles = ["" for _ in range(b)]

    axes = axes.flatten()

    assert len(titles) == b
    assert b <= axes.size

    for image, axis, title in zip(_image_array, axes, titles):
        axis.imshow(image, cmap='gray')
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        axis.set_title(title, fontsize=8)

    # 将 plot 保存为图像，还是直接显示
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        from datetime import datetime
        _title = save_title if save_title is not None else "no title"
        save_path = os.path.join(save_dir, _title + "_" + datetime.now().strftime(r"%m_%d %H_%M_%S") + ".png")
        pyplot.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        pyplot.show()


def plot_arrays(x: 'torch.Tensor', ys: 'torch.Tensor'):
    _, axes = pyplot.subplots(1, 1)
    for y in ys:
        axes.plot(x.cpu().numpy(), y.cpu().numpy(), markersize=3)

    pyplot.show()