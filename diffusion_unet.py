from diffusers.models.unets.unet_2d import UNet2DModel
import torch 
import torch.nn as nn
from typing import Optional

class DFUNet(nn.Module):
    """
        Hugging face 的 专用于扩散模型的Unet，添加了条件输入通道链接条件信息
            Args:
                ch_input: 输入通道数
                ch_output: 输出通道数  
                image_size: 图像尺寸
                condition_channels: 条件通道数
                layers: 每个块的层数，默认2
                base_channels: 基础通道数，默认64
                channel_multipliers: 通道倍数元组，默认(1,1,2,2,4,4)
                num_blocks: 块的数量，默认6
                
            Raises:
                ValueError: 当参数无效时抛出异常
    """
    def __init__(self,
             ch_input: int,
             ch_output: int,
             image_size: int,
             condition_channels: int = 0,
             layers: int = 2,
             base_channels: int = 64,
             channel_multipliers: tuple = (1, 1, 2, 2, 4, 4),
             num_blocks: int = 6
             ) -> None:
        
        # 计算实际输入通道数： 原输入 + 条件输入
        actual_input_channels = ch_input + condition_channels

        # 动态计算通道配置
        block_out_channels = tuple(base_channels * mult for mult in channel_multipliers)
        
        # 生成块类型配置
        down_block_types = tuple("DownBlock2D" for _ in range(num_blocks))
        up_block_types = tuple("UpBlock2D" for _ in range(num_blocks))

        super().__init__()  # 确保父类的 __init__ 方法被正确地调用

        self.condition_channels = condition_channels

        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=actual_input_channels,
            out_channels=ch_output,
            layers_per_block=layers,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types
        )
    def forward(self, x:torch.Tensor, ts:torch.Tensor,condition:Optional[torch.Tensor] = None) -> torch.Tensor:
        if condition is not None and self.condition_channels > 0:
            x = torch.cat([x, condition], dim=1)
        return self.model(x, ts).sample
    
if __name__ == '__main__':
    dnf_unet = DFUNet(
        ch_input=1,
        ch_output=1,
        image_size=256,
        layers=2,
        base_channels=64,
        channel_multipliers=(1, 1, 2, 2, 4, 4),
        num_blocks=6
    )
    # 测试DNFUnet，这的batch_size必须和time step 保持一致
    x = torch.randn(3, 1, 256, 256)
    ts = torch.randn([3])
    out = dnf_unet.forward(x, ts)
    print(x)
    print(out)
