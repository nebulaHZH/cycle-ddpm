import torch
from tqdm import tqdm
from cycle_ddpm import Configs, CycleDDPM
from torch.utils.data import DataLoader
from load import PairedImageDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

config = Configs(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dir_a_path="D:/0-nebula/dataset/ixi_paried/t1_30_resized",
    dir_b_path="D:/0-nebula/dataset/ixi_paried/t2_30_resized",
    image_size=256,
    batch=1,
    epochs=20,
    lr=1e-4,
    clip=1.0,
    num_train_timesteps=1000,
    num_inference_timesteps=20,
    beta_start=0.0001,
    beta_end=0.005,
)

if __name__ == "__main__":
    model = CycleDDPM(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler_lr = CosineAnnealingLR(optimizer, T_max=config.epochs)

    paired_dataset = PairedImageDataset(dir_A=config.dir_a_path,dir_B=config.dir_b_path)


    data_loader = DataLoader(paired_dataset, batch_size=config.batch, shuffle=False)

    loss = {}
    for epoch in range(config.epochs):
        total_steps = len(data_loader)
        progress_bar = tqdm(data_loader, total=total_steps,desc=f"Epoch {epoch+1}/{config.epochs}")
        model.train()
        avg_loss = 0
        avg_cycle_loss = 0
        avg_a_losss = 0
        avg_b_losss = 0
        count = 0
        for batch_data in progress_bar:
            optimizer.zero_grad()
            batch_A,batch_B = batch_data # 从配对数据中解包
            real_A = batch_A.to(config.device)  # CT图像
            real_B = batch_B.to(config.device)  # MRI图像
            if epoch >= config.epochs * 3 // 4:
                loss = model.compute_loss(real_A, real_B,0.1,epoch,config.epochs)
            else:
                loss = model.compute_loss(real_A, real_B,0,epoch)
            loss['loss'].backward()
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            avg_loss = avg_loss + loss['loss'].detach().item()
            avg_a_losss = avg_a_losss + loss['loss_A'].detach().item()
            avg_b_losss = avg_b_losss + loss['loss_B'].detach().item()
            avg_cycle_loss = avg_cycle_loss + loss['cycle_loss'].detach().item()
            count = count + 1
            logs = {
                "Loss": f"{avg_loss / count:.4f}",
                "Cycle_Loss": f"{avg_cycle_loss / count:.4f}",
                "Loss_A": f"{avg_a_losss / count:.4f}",
                "Loss_B": f"{avg_b_losss / count:.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            }
            progress_bar.set_postfix(logs)
        scheduler_lr.step()

    # 保存模型
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, r"D:/0-nebula/dataset/checkpoints/" + str(config.epochs))

