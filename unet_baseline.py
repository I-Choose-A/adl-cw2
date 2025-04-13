import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data.dataset import OxfordIIITPet
from models.unet import UNet
from utils.loss import weighted_loss
from utils.mask_utils import get_trimap

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
        tqdm.write(f"EPOCH: {epoch + 1}/{epochs}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_iou: {val_iou:.4f}")

    torch.save(model.state_dict(), "models/unet_supervised.pth")

def test_unet(model):
    model.eval()
    model = model.to(device)

    test_loss = 0.
    test_iou = 0.
    for x, _, image_ids in test_loader:
        x = x.to(device)
        mask = get_trimap(image_ids).to(device)

        with torch.no_grad():
            pred_mask = model(x)
            loss = weighted_loss(pred_mask, mask)
            test_loss += loss.item() * x.shape[0]

            pred_binary = (pred_mask > 0.5).float()
            intersection = (pred_binary * mask).sum((1, 2, 3))
            union = (pred_binary + mask).clamp(0, 1).sum((1, 2, 3))
            batch_iou = (intersection / (union + 1e-6)).sum().item()
            test_iou += batch_iou

    test_loss /= len(test_dataset)
    test_iou /= len(test_dataset)

    print(f"Test loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}")

if __name__ == '__main__':
    if not os.path.exists("models/unet_supervised.pth"):
        train_unet_supervised(unet)
    else:
        unet.load_state_dict(torch.load("models/unet_supervised.pth"))