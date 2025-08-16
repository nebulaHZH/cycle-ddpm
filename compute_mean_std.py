from torch.utils.data import DataLoader
from load import PairedImageDataset

# 计算整个数据集的真实均值和标准差
def calculate_mean_std(dt: PairedImageDataset):
    loader = DataLoader(dt, batch_size=1, shuffle=False)
    m = 0.0
    s = 0.0
    total_samples = 0
    for data_A, data_B in loader:
        batch_samples = data_A.size(0)
        data_A = data_A.view(batch_samples, data_A.size(1), -1)
        m += data_A.mean(2).sum(0)
        s += data_A.std(2).sum(0)
        total_samples += batch_samples

    m /= total_samples
    s /= total_samples
    return m, s


if __name__ == "__main__":

    dataset = PairedImageDataset(dir_A="D:/0-nebula/dataset/ixi_paried/t1_30_resized",
                                 dir_B="D:/0-nebula/dataset/ixi_paried/t2_30_resized",
                                 image_size=256,
                                 gray_scale=True)
    mean, std = calculate_mean_std(dataset)
    print(f"Mean: {mean}", f"Std: {std}")