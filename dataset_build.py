"""
读取文件路径并把文件夹下的子文件夹中所有名为slice_160，slice_165，slice_170的文件放在一个文件夹下
"""


import os
import shutil

def copy_target_slices(source_root, target_dir):
    """
    从源目录的子文件夹中提取指定切片文件并集中到目标目录
    参数:
        source_root (str): 需要遍历的源目录路径
        target_dir (str): 目标存储目录路径
    """
    # 确保输出目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 定义需要提取的文件名集合
    target_files = {'slice_065.png', 'slice_070.png', 'slice_075.png'}
    
    i = 0
    # 遍历目录树
    for root, _, files in os.walk(source_root):
        for file in files:
            if file in target_files:
                src_path = os.path.join(root, file)
                
                # 获取相对路径以创建唯一文件名
                relative_path = os.path.relpath(root, source_root)
                # 将路径分隔符替换为下划线，避免文件名中的非法字符
                path_prefix = relative_path.replace(os.path.sep, '_').replace('.', '')
                
                # 构造新文件名：路径前缀_原文件名
                if path_prefix:
                    new_filename = f"{path_prefix}_{file}"
                else:
                    new_filename = file
                    
                dst_path = os.path.join(target_dir, new_filename)
                
                try:
                    # 复制文件并保留元数据
                    shutil.copy2(src_path, dst_path)
                    print(f"成功复制: {src_path} -> {dst_path}")
                except Exception as e:
                    print(f"复制失败 {src_path}: {str(e)}")

if __name__ == "__main__":
    # 源目录和目标目录设置
    source_path = r"D:\\0-nebula\\dataset\\ixi_unzip\\t2_png"
    target_path = r"D:\\0-nebula\\dataset\\ixi_unzip\\t2_png_train"
    
    # 执行文件复制
    copy_target_slices(source_path, target_path)