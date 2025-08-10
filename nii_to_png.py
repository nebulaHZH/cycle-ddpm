"""
nii 三维图像切片以水平面的角度转化为多个 png
    将nii图像中后5个到前5个切片转化为png
输入：
    nii_path: nii文件夹路径
    png_path: png文件夹保存路径
"""

import os
from nibabel.loadsave import load as nib_load  # 显式导入避免Pylance误报
from nibabel.funcs import as_closest_canonical
from scipy.ndimage import rotate,zoom
import numpy as np
from PIL import Image
import os

def get_file_path(nii_path:str) -> list:
    file_path = [os.path.join(nii_path, file) for file in os.listdir(nii_path) if file.endswith(".nii")]
    return file_path
def adjust_image_ratio(slice_data, zooms):
    # 获取 X, Y 方向的像素间距
    zoom_x, zoom_y = zooms[0], zooms[1]
    
    # 计算调整比例
    ratio = zoom_x / zoom_y
    
    # 调整图像比例
    if ratio != 1:
        new_shape = (int(slice_data.shape[0] * ratio), slice_data.shape[1])
        slice_data = zoom(slice_data, (ratio, 1), order=1)
    
    return slice_data

def nii_to_png(nii_list:list,png_path:str) -> None:
    for nii_path in nii_list:
        # 读取nii文件
        nii_data = nib_load(nii_path)
        
        nii_data = as_closest_canonical(nii_data)  # 确保方向正确
        image_array = nii_data.get_fdata() # type: ignore
        print(f"Image shape: {image_array.shape}") # 256,256,150

        zooms = nii_data.header.get_zooms()
        print(f"Pixel spacing: {zooms}")

        # 获取文件名（不包含扩展名）作为文件夹名
        filename = os.path.splitext(os.path.basename(nii_path))[0]
        subject_folder = os.path.join(png_path, filename)
        
        # 创建对应的文件夹
        os.makedirs(subject_folder, exist_ok=True)
        
        # 获取图像的深度维度（通常是第三个维度）
        depth = image_array.shape[2]
        
        # 将中间部分的切片转化为png（从前后各5个切片开始到结束）
        start_slice = 5
        end_slice = depth - 5 
        
        for i in range(start_slice, min(end_slice, depth),5):
            # 获取单个切片
            slice_data = image_array[: ,:,i]
            # 调整图像比例
            slice_data = adjust_image_ratio(slice_data, zooms[:2])
            # 旋转图像
            slice_data = rotate(slice_data, angle=90)  # 旋转90度
            # 标准化数据到0-255范围
            slice_data = ((slice_data - np.min(slice_data)) / 
                         (np.max(slice_data) - np.min(slice_data)) * 255).astype(np.uint8)
            
            # 转换为PIL图像并保存
            img = Image.fromarray(slice_data)
            # img = img.resize((256, 256), Image.Resampling.BILINEAR)
            img.save(os.path.join(subject_folder, f"slice_{i:03d}.png"))


nii_t1_path = "D:/0-nebula/dataset/ixi_unzip/t1_ixi"
nii_t2_path = "D:/0-nebula/dataset/ixi_unzip/t2_ixi"
nii_to_png(get_file_path(nii_t1_path), "D:/0-nebula/dataset/ixi_unzip/t1_png")
nii_to_png(get_file_path(nii_t2_path), "D:/0-nebula/dataset/ixi_unzip/t2_png")