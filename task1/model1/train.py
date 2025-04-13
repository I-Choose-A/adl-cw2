import sys
import os
import datetime
from PIL import Image

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from task1.data.dataset import OxfordIIITPet
from models.unet import UNet
from utils.loss import weighted_loss
from utils.mask_utils import get_trimap

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
torch.manual_seed(2025)
batch_size = 32

# Load dataset
dataset = OxfordIIITPet()
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

unet = UNet()


def train_unet_supervised(model):
    tqdm.write("Start training fully supervised UNet...")
    epochs = 10
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-6)
    model = model.to(device)

    for epoch in range(epochs):
        train_loss = 0.
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for x, _, image_ids in train_bar:
            x = x.to(device)
            mask = get_trimap(image_ids).to(device)

            optimizer.zero_grad()
            pred_mask = model(x)
            loss = weighted_loss(pred_mask, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.shape[0]
            train_bar.set_postfix({"loss": loss.item()})
        train_loss /= len(train_dataset)

        # Evaluation
        model.eval()
        val_loss = 0.
        val_iou = 0.
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
        for x, _, image_ids in val_bar:
            x = x.to(device)
            mask = get_trimap(image_ids).to(device)

            with torch.no_grad():
                pred_mask = model(x)
                loss = weighted_loss(pred_mask, mask)
                val_loss += loss.item() * x.shape[0]

                pred_binary = (pred_mask > 0.5).float()
                intersection = (pred_binary * mask).sum((1, 2, 3))
                union = (pred_binary + mask).clamp(0, 1).sum((1, 2, 3))
                batch_iou = (intersection / (union + 1e-6)).sum().item()
                val_iou += batch_iou

                val_bar.set_postfix({"loss": loss.item()})

        val_loss /= len(val_dataset)
        val_iou /= len(val_dataset)

        train_bar.clear()
        val_bar.clear()
        tqdm.write(f"EPOCH: {epoch + 1}/{epochs}, train_loss: {train_loss:.4f}, "
                   f"val_loss: {val_loss:.4f}, val_iou: {val_iou:.4f}")

    torch.save(model.state_dict(), "task1/model1/models/unet_supervised.pth")


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.clone()
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor.clamp_(0, 1)


def test_unet(model):
    model = model.to(device)
    print(f"Start testing UNet model... {datetime.datetime.now()}")
    # print("Using device:", device)
    model.eval()

    test_loss = 0.
    test_iou = 0.
    for x, _, image_ids in test_loader:
        x = x.to(device)
        trimap = get_trimap(image_ids).to(device)

        with torch.no_grad():
            pred_mask = model(x)
            loss = weighted_loss(pred_mask, trimap)
            test_loss += loss.item()

            pred_binary = (pred_mask > 0.5).float()
            intersection = (pred_binary * trimap).sum((1, 2, 3))
            union = (pred_binary + trimap).clamp(0, 1).sum((1, 2, 3))
            batch_iou = (intersection / (union + 1e-6)).sum().item()
            test_iou += batch_iou

    test_loss /= len(test_dataset)
    test_iou /= len(test_dataset)

    print(f"Test loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}, {datetime.datetime.now()}")

    # visual sample
    model.eval()
    for i, (x, _, image_ids) in enumerate(test_loader):
        x = x.to(device)

        # original photo
        x_denorm = denormalize(x[0].unsqueeze(0).to('cpu'))
        image_np = x_denorm.squeeze(0).permute(1, 2, 0).numpy()
        image_uint8 = (image_np * 255).astype(np.uint8)
        Image.fromarray(image_uint8).show()

        # Ground truth mask
        trimap = get_trimap(image_ids)
        gt_mask = (trimap[0].squeeze(0) > 0.5).float() * 255
        Image.fromarray(gt_mask.to('cpu').numpy().astype(np.uint8), mode='L').show()

        # Prediction
        with torch.no_grad():
            pred_mask = model(x)
        pred_binary = (pred_mask[0].squeeze(0) > 0.5).float() * 255
        Image.fromarray(pred_binary.to('cpu').numpy().astype(np.uint8), mode='L').show()

        break


if __name__ == '__main__':
    if not os.path.exists("task1/model1/models/unet_supervised.pth"):
        train_unet_supervised(unet)
    else:
        unet.load_state_dict(torch.load("task1/model1/models/unet_supervised.pth", weights_only=True))

    test_unet(unet)