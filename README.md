# Cycle-DDPM
    添加了循环一致性损失得DDPM模型

## 项目结构
    - train_ddpm.py 训练该模型，基础配置文件
    - cycle_ddpm.py 加入了cycle_loss的DDPM模型
    - diffusion_unet.py 用于DDPM的Unet模型，采用的Hugging face的专用于扩散模型的Unet
    - plot.py 用于画图
    - scehedule.py 设置Diffusion model 基本的相关函数
    - load.py 用于加载数据集，数据集的格式为
        - t1:
            - 0001.png
            - 0002.png
            - ......
        - t2:
            - 0001.png
            - 0002.png
            - ......
    - test.py 用于测试模型的转换和生成效果

## 使用方法
1. 配置数据集
   - 数据集的格式为png
2. 安装依赖
    ```bash
    pip install -r requirements.txt
   ```
3. 训练模型
   - 运行train_ddpm.py 
4. 测试模型
   - 运行test.py

## 进度说明

### 2025.8.10
    1. 上传该项目到github
    2. 修改了lambda_cycle 的 权重为：前epochs//2 为 0 ，后epochs//2 为 1
    3. 效果很差
#### 效果图
    1. 在计算循环一致性损失时的生成图对比：
![img.png](examples/2025_8_10_1.png)

    2. 循环损失一直无法下降，连两个ddpm模型的损失也降不下去
![img.png](examples/2025_8_10_2.png)

### 2025.8.11
1. 将训练集样本个数降低到30个
2. 将epoch提升至200个

#### 效果图
    1. 训练后，单个模型的损失降低到0.01左右，但是循环一致性损失还是很高
![img.png](examples/2025_8_11_2.png)

    2. 但是测试集发现效果已经出现了
![img.png](examples/2025_8_11_1.png)

### 2025.8.12
    1. 将循环一致性损失的权重由1设置为0.1，正在训练中...