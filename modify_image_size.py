# 把D:\0-nebula\dataset\ixi_paried\t1文件夹下的png图片尺寸从192x256才用两边填充黑边的方式修改为256x256，保持中间不同

import os
import numpy as np
from PIL import Image
import nibabel as nib

def modify_image_size():
    # 定义输入和输出文件夹路径
    input_folder = r"D:\\0-nebula\\dataset\\ixi_paried\\t1"
    output_folder = r"D:\\0-nebula\\dataset\\ixi_paried\\t1_resized"
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有PNG文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # 构建完整的文件路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # 打开图像
            img = Image.open(input_path)
            
            # 检查图像尺寸是否为192x256
            if img.size == (192, 256):
                # 创建一个新的256x256的黑色背景图像
                new_img = Image.new('RGB', (256, 256), (0, 0, 0))
                
                # 计算在新图像中居中放置的位置
                # 在宽度方向上需要填充(256-192)/2 = 32像素，每边16像素
                # 高度方向保持不变
                x_offset = (256 - 192) // 2
                y_offset = 0
                
                # 将原图像粘贴到新图像的中央
                new_img.paste(img, (x_offset, y_offset))
                
                # 保存修改后的图像
                new_img.save(output_path)
                print(f"已处理: {filename}")
            else:
                print(f"警告: {filename} 的尺寸不是192x256，跳过处理")
    
    print("所有图像处理完成")

# 执行函数
if __name__ == "__main__":
    modify_image_size()