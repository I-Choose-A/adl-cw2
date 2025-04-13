import datetime
import torch
from torch.utils.data import DataLoader, random_split

from data.dataset_bbox import OxfordIIITPet
from models.unet import UNet
from utils.loss import weighted_loss
from utils.mask_utils import get_trimap,get_cam
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(2025)
batch_size = 32


def custom_collate_fn(batch):
    images, labels, image_ids, bboxes = zip(*batch)
    return torch.stack(images), labels, list(image_ids), list(bboxes)


dataset = OxfordIIITPet()
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                          pin_memory=True, collate_fn=custom_collate_fn)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=True, collate_fn=custom_collate_fn)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                         pin_memory=True, collate_fn=custom_collate_fn)

# 定义模型
unet = UNet().to(device)

def train_unet(model, model_name, save_to = '',noise_ratio = 0):
    print(f"start training UNet, {datetime.datetime.now()}")

    epochs = 20
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4, weight_decay=1e-5)
    model = model.to(device)
    best_iou = 0.0

    for epoch in range(epochs):
        train_loss = 0.
        for i, (x, _, image_ids, _) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()
            cam = get_cam(image_ids)

            # if epoch == 0:
            #     mask = cam
            # elif epoch < 3:
            #     mask = 0.8 * cam + 0.2 * model(x).detach()
            # else:
            #     alpha = min(0.3 * (epoch - 3), 0.6)
            #     mask = (1 - alpha) * cam + alpha * model(x).detach()

            mask = (cam>0.3).float()

            mask = corrupt_mask(mask, corruption_prob=noise_ratio)

            pred_mask = model(x)
            loss = weighted_loss(pred_mask, mask)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataset)

        # evaluation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for x, _, image_ids, _ in val_loader:
                x = x.to(device)
                trimap = get_trimap(image_ids).to(device)
                pred_mask = model(x)
                loss = weighted_loss(pred_mask, trimap)
                val_loss += loss.item() * x.size(0)

                pred_binary = (pred_mask > 0.3).float()
                intersection = (pred_binary * trimap).sum((1, 2, 3))
                union = (pred_binary + trimap).clamp(0, 1).sum((1, 2, 3))
                val_iou += (intersection / (union + 1e-6)).sum().item()

        val_loss /= len(val_dataset)
        val_iou /= len(val_dataset)

        print(f"[{model_name}] Epoch {epoch+1} | TrainLoss={train_loss/len(train_loader):.4f} | ValLoss={val_loss:.4f} | ValIoU={val_iou:.4f}")

        # 保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), f"models/best_{model_name}")
            print(f"[✓] Saved best model at epoch {epoch+1} with IoU={best_iou:.4f}")

    torch.save(model.state_dict(), save_to)

def corrupt_mask(mask, corruption_prob=0):
    '''
    用来给mask加噪声,仅用于生成不同质量的伪标签
    '''
    rand_noise = torch.rand_like(mask)
    flip_mask = (rand_noise < corruption_prob).float()
    corrupted = torch.abs(mask - flip_mask)  # flip: 1->0, 0->1
    return corrupted

def evaluate_model(unet, test_loader, device):
    unet.eval()
    total_iou = 0.
    total_loss = 0.
    cam_iou = 0.
    cam_iou_5 = 0.
    cam_iou_10 = 0.

    with torch.no_grad():
        for x, _, image_ids, _ in test_loader:
            x = x.to(device)
            trimap = get_trimap(image_ids).to(device)
            true_mask = trimap

            # 模型预测
            pred_mask = unet(x)
            loss = weighted_loss(pred_mask, true_mask)
            total_loss += loss.item()

            pred_binary = (pred_mask > 0.3).float()
            intersection = (pred_binary * true_mask).sum((1, 2, 3))
            union = (pred_binary + true_mask).clamp(0, 1).sum((1, 2, 3))
            batch_iou = (intersection / (union + 1e-6)).sum().item()
            total_iou += batch_iou

            # CAM 原始与噪声版本
            cam = get_cam(image_ids).to(device)
            cam_bin = (cam > 0.3).float()

            cam_noise_5 = corrupt_mask(cam_bin, corruption_prob=0.05)
            cam_noise_10 = corrupt_mask(cam_bin, corruption_prob=0.10)

            cam_iou += ((cam_bin * trimap).sum((1, 2, 3)) /
                        ((cam_bin + trimap).clamp(0, 1).sum((1, 2, 3)) + 1e-6)).sum().item()

            cam_iou_5 += ((cam_noise_5 * trimap).sum((1, 2, 3)) /
                          ((cam_noise_5 + trimap).clamp(0, 1).sum((1, 2, 3)) + 1e-6)).sum().item()

            cam_iou_10 += ((cam_noise_10 * trimap).sum((1, 2, 3)) /
                           ((cam_noise_10 + trimap).clamp(0, 1).sum((1, 2, 3)) + 1e-6)).sum().item()

    n = len(test_loader.dataset)
    avg_loss = total_loss / n
    avg_iou = total_iou / n
    avg_cam_iou = cam_iou / n
    avg_cam_iou_5 = cam_iou_5 / n
    avg_cam_iou_10 = cam_iou_10 / n

    print(f"Test Loss: {avg_loss:.4f}, Test IoU: {avg_iou:.4f}")
    print(f"Raw CAM IoU:      {avg_cam_iou:.4f}")
    print(f"Noise 5% CAM IoU: {avg_cam_iou_5:.4f}")
    print(f"Noise 10% CAM IoU:{avg_cam_iou_10:.4f}")

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.clone()
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor.clamp_(0, 1)

def visualize_set(trimap_tensor, cam, cam_5, cam_10,
                  base_pred, best_5_pred, best_10_pred, image_id,
                  save_dir="outputs_noise"):
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)

    def to_np(tensor):
        return tensor.squeeze().detach().cpu().numpy()

    def to_bin_np(tensor):
        return (to_np(tensor) > 0.3).astype(float)

    # 图像数据
    trimap_np = to_np(trimap_tensor)
    cam_np = to_np(cam)
    cam_5_np = to_np(cam_5)
    cam_10_np = to_np(cam_10)
    base_np = to_bin_np(base_pred)
    best_5_np = to_bin_np(best_5_pred)
    best_10_np = to_bin_np(best_10_pred)

    titles = [
        "Trimap", "Raw CAM", "CAM w/ 5% Noise",
        "CAM w/ 10% Noise", "Best Unet Prediction",
        "Best 5% Prediction", "Best 10% Prediction"
    ]
    images = [
        trimap_np, cam_np, cam_5_np,
        cam_10_np, base_np,
        best_5_np, best_10_np
    ]

    fig, axs = plt.subplots(3, 3, figsize=(12, 10))

    for i in range(len(images)):
        ax = axs.flat[i]
        ax.imshow(images[i], cmap="gray")
        ax.set_title(titles[i])
        ax.axis("off")

    # 隐藏多余 subplot
    for j in range(len(images), 9):
        axs.flat[j].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{image_id}_comparison.png")
    plt.savefig(save_path)
    plt.close()


    
if __name__ == '__main__':
    
    dirty_5_model = UNet().to(device)
    dirty_10_model = UNet().to(device)
    
    # 5%污染伪像素
    if not os.path.exists("models/unet_dirty_5.pth"):
        train_unet(dirty_5_model,model_name="unet_dirty_5.pth", save_to="models/unet_dirty_5.pth",noise_ratio=0.05)
    else:
        dirty_5_model.load_state_dict(torch.load("models/unet_dirty_5.pth"))

    # 10%污染伪像素
    if not os.path.exists("models/unet_dirty_10.pth"):
        train_unet(dirty_10_model,model_name="unet_dirty_10.pth", save_to="models/unet_dirty_10.pth",noise_ratio=0.1)
    else:
        dirty_10_model.load_state_dict(torch.load("models/unet_dirty_10.pth"))
    
    best_5_model = UNet().to(device)
    best_10_model = UNet().to(device)
    
    if os.path.exists("models/best_unet_nobbox.pth"):
        base_model = UNet().to(device)
        base_model.load_state_dict(torch.load("models/best_unet_nobbox.pth", map_location=device))
        evaluate_model(base_model, test_loader,device=device)
    else:
        print("best_unet_nobbox.pth not found.")
        
    if os.path.exists("models/best_unet_dirty_5.pth"):
        best_5_model.load_state_dict(torch.load("models/best_unet_dirty_5.pth", map_location=device))
        best_5_model.eval()
        print("Dirty model trained on 5% noisy pseudo label evaluated on clean labels):")
        evaluate_model(best_5_model,test_loader=test_loader,device=device)
    else:
        print("best_unet_dirty_5.pth not found.")

    if os.path.exists("models/best_unet_dirty_10.pth"):
        best_10_model.load_state_dict(torch.load("models/best_unet_dirty_10.pth", map_location=device))
        best_10_model.eval()
        print("Dirty model trained on 10% noisy pseudo label evaluated on clean labels):")
        evaluate_model(best_10_model,test_loader=test_loader,device=device)
    else:
        print("best_unet_dirty_10.pth not found.")

    x_batch, _, image_ids, _ = next(iter(test_loader))
    x_batch = x_batch.to(device)
    cam_batch = get_cam(image_ids).to(device)
    trimap_batch = get_trimap(image_ids).to(device)

    # 模型推理
    with torch.no_grad():
        base_preds = base_model(x_batch)  # 新增
        best_5_preds = best_5_model(x_batch)
        best_10_preds = best_10_model(x_batch)

    # 生成图像对比
    for i in range(5):
        cam = (cam_batch[i:i+1] > 0.3).float()
        cam_5 = corrupt_mask(cam, corruption_prob=0.05)
        cam_10 = corrupt_mask(cam, corruption_prob=0.1)

        visualize_set(
            trimap_batch[i],
            cam[0].cpu(),
            cam_5[0].cpu(),
            cam_10[0].cpu(),
            base_preds[i].cpu(),          # ✅ 新增 base model 输出
            best_5_preds[i].cpu(),
            best_10_preds[i].cpu(),
            image_ids[i]
        )


