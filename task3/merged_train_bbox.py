# ======== 设置与导入 ========
import os
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import matplotlib.patches as patches

from data.dataset_bbox import OxfordIIITPet
from eval import eval_classifier
from models.resnet import ResNet18, ResNet50
from models.unet import UNet
from utils.loss import weighted_loss
from utils.mask_utils import create_cam, get_cam, get_trimap

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2025)
batch_size = 32

# ======== 工具函数 ========
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.clone()
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor.clamp_(0, 1)

def custom_collate_fn(batch):
    images, labels, image_ids, bboxes = zip(*batch)
    return torch.stack(images), torch.tensor(labels), list(image_ids), list(bboxes)

class NoBBoxWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, image_id, _ = self.dataset[idx]
        return image, label, image_id

# ======== 模型训练 ========
def train_classifier(model):
    epochs = 20
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_acc = 0.0  # 可选：保存最优模型

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0., 0.
        for x, y, _ in train_loader_classifier:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (pred.argmax(1) == y).sum().item()
        train_acc = total_correct / len(train_dataset)

        # ========== 验证 ========== #
        model.eval()
        val_correct = 0.
        val_loss = 0.
        with torch.no_grad():
            for x, y, _ in val_loader_classifier:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item() * x.size(0)
                val_correct += (pred.argmax(1) == y).sum().item()
        val_acc = val_correct / len(val_dataset)
        val_loss /= len(val_dataset)

        print(f"[Classifier] Epoch {epoch+1} | Train Acc={train_acc*100:.2f}% | Val Acc={val_acc*100:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_resnet18.pth")

    torch.save(model.state_dict(), "models/resnet18.pth")

def train_unet(model, use_bbox=True, model_name="unet_bbox.pth"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    model.to(device)
    epochs = 20

    best_iou = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.
        for i, (x, _, image_ids, bboxes) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            cam = get_cam(image_ids).to(device)
            _, _, H, W = cam.shape

            if use_bbox:
                for j in range(cam.shape[0]):
                    bbox = bboxes[j]
                    if bbox:
                        xmin, ymin, xmax, ymax = map(int, bbox)
                        if 0 <= xmin < xmax <= W and 0 <= ymin < ymax <= H:
                            mask = torch.zeros((H, W), device=device)
                            mask[ymin:ymax, xmin:xmax] = 1.
                            cam[j, 0] = cam[j, 0].float() * mask

            mask = (cam > 0.3).float()
            pred = model(x)
            loss = weighted_loss(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

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

        print(f"[{model_name}] Epoch {epoch+1} | TrainLoss={running_loss/len(train_loader):.4f} | ValLoss={val_loss:.4f} | ValIoU={val_iou:.4f}")

        # 保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), f"models/best_{model_name}")
            print(f"[✓] Saved best model at epoch {epoch+1} with IoU={best_iou:.4f}")

    torch.save(model.state_dict(), f"models/{model_name}")

# ======== 模型评估 ========
def evaluate_unet(model, loader, label=""):
    model.eval()
    model.to(device)
    total_iou, total_loss = 0., 0.
    cam_iou, bbox_cam_iou = 0., 0.

    with torch.no_grad():
        for x, _, image_ids, bboxes in tqdm(loader):
            x = x.to(device)
            trimap = get_trimap(image_ids).to(device)

            # 模型预测
            pred = model(x)
            loss = weighted_loss(pred, trimap)
            total_loss += loss.item()

            pred_bin = (pred > 0.3).float()
            intersection = (pred_bin * trimap).sum((1, 2, 3))
            union = (pred_bin + trimap).clamp(0, 1).sum((1, 2, 3))
            total_iou += (intersection / (union + 1e-6)).sum().item()

            # 原始 CAM
            cam = get_cam(image_ids).to(device)
            cam_bin = (cam > 0.3).float()
            cam_iou += ((cam_bin * trimap).sum((1, 2, 3)) /
                        ((cam_bin + trimap).clamp(0, 1).sum((1, 2, 3)) + 1e-6)).sum().item()

            # 加 bbox 的 CAM
            masked_cam = cam.clone()
            _, _, H, W = masked_cam.shape
            for i in range(masked_cam.shape[0]):
                bbox = bboxes[i]
                if bbox:
                    xmin, ymin, xmax, ymax = map(int, bbox)
                    if 0 <= xmin < xmax <= W and 0 <= ymin < ymax <= H:
                        mask = torch.zeros((H, W), device=device)
                        mask[ymin:ymax, xmin:xmax] = 1.0
                        masked_cam[i, 0] = masked_cam[i, 0].float() * mask
            masked_bin = (masked_cam > 0.3).float()
            bbox_cam_iou += ((masked_bin * trimap).sum((1, 2, 3)) /
                             ((masked_bin + trimap).clamp(0, 1).sum((1, 2, 3)) + 1e-6)).sum().item()

    n = len(loader.dataset)
    print(f"[{label}] Test Loss={total_loss/n:.4f}, IoU={total_iou/n:.4f}, "
          f"CAM-IoU={cam_iou/n:.4f}, BBox-CAM-IoU={bbox_cam_iou/n:.4f}")

    

# ======== 主流程 ========
if __name__ == "__main__":
    # ======== 数据准备 ========
    dataset = OxfordIIITPet()
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    train_loader_classifier = DataLoader(NoBBoxWrapper(train_dataset), batch_size=batch_size, shuffle=True)
    val_loader_classifier = DataLoader(NoBBoxWrapper(val_dataset), batch_size=batch_size, shuffle=False)
    
    resnet = ResNet50
    unet_bbox = UNet()
    unet_nobbox = UNet()

    if not os.path.exists("models/resnet18.pth"):
        train_classifier(resnet)
    else:
        resnet.load_state_dict(torch.load("models/resnet18.pth", map_location=device))

    if not os.path.exists("data/CAM"):
        for loader in [train_loader, val_loader, test_loader]:
            for x, y, ids in DataLoader(NoBBoxWrapper(loader.dataset), batch_size=batch_size):
                create_cam(resnet, x.to(device), y.to(device), ids)

    if not os.path.exists("models/unet_bbox.pth"):
        train_unet(unet_bbox, use_bbox=True, model_name="unet_bbox.pth")
    else:
        unet_bbox.load_state_dict(torch.load("models/unet_bbox.pth"))

    if not os.path.exists("models/unet_nobbox.pth"):
        train_unet(unet_nobbox, use_bbox=False, model_name="unet_nobbox.pth")
    else:
        unet_nobbox.load_state_dict(torch.load("models/unet_nobbox.pth"))

    # ========== 加载并评估 Best 模型 ==========

    if os.path.exists("models/best_unet_bbox.pth"):
        best_unet_bbox = UNet().to(device)
        best_unet_bbox.load_state_dict(torch.load("models/best_unet_bbox.pth", map_location=device))
        evaluate_unet(best_unet_bbox, test_loader, label="BBox (Best)")
    else:
        print("best_unet_bbox.pth not found.")

    if os.path.exists("models/best_unet_nobbox.pth"):
        best_unet_nobbox = UNet().to(device)
        best_unet_nobbox.load_state_dict(torch.load("models/best_unet_nobbox.pth", map_location=device))
        evaluate_unet(best_unet_nobbox, test_loader, label="NoBBox (Best)")
    else:
        print("best_unet_nobbox.pth not found.")
        