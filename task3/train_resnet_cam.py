import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data.dataset_bbox import OxfordIIITPet  # Default with bbox, we'll wrap it below
from models.resnet import ResNet18
from eval import eval_classifier
from utils.mask_utils import create_cam, get_trimap, get_cam

from utils.loss import weighted_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2025)
batch_size = 32


# ====== pack to no bbox Dataset ======
class NoBBoxWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, image_id, _ = self.dataset[idx]
        return image, label, image_id


def get_resnet_model(name):
    if name == "resnet18":
        from models.resnet import ResNet18

        return ResNet18
    elif name == "resnet34":
        from models.resnet import ResNet34

        return ResNet34
    elif name == "resnet50":
        from models.resnet import ResNet50

        return ResNet50
    else:
        raise ValueError(f"Unsupported model: {name}")


# ====== ResNet train ======
def train_classifier(model, train_loader, val_loader, train_dataset, val_dataset):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0.0, 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (pred.argmax(1) == y).sum().item()

        train_acc = total_correct / len(train_dataset)

        model.eval()
        val_correct, val_loss = 0.0, 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (pred.argmax(1) == y).sum().item()

        val_acc = val_correct / len(val_dataset)
        print(
            f"[ResNet] Epoch {epoch+1} | Train Acc={train_acc*100:.2f}% | Val Acc={val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_resnet18.pth")

    torch.save(model.state_dict(), "models/resnet18.pth")


def evaluate_cam_iou(loader, cam_thresh=0.5):
    total_iou = 0.0
    for x, _, image_ids in tqdm(loader, desc="Evaluating CAM IoU"):
        trimap = get_trimap(image_ids).to(device)
        cam = get_cam(image_ids).to(device)
        cam_bin = (cam > cam_thresh).float()

        iou = (cam_bin * trimap).sum((1, 2, 3)) / (
            (cam_bin + trimap).clamp(0, 1).sum((1, 2, 3)) + 1e-6
        )
        total_iou += iou.sum().item()

    avg_iou = total_iou / len(loader.dataset)
    print(f"[CAM] Avg IoU with Trimap: {avg_iou:.4f}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    dataset_full = OxfordIIITPet()
    train_size = int(len(dataset_full) * 0.8)
    val_size = int(len(dataset_full) * 0.1)
    test_size = len(dataset_full) - train_size - val_size
    train_set, val_set, test_set = random_split(
        dataset_full, [train_size, val_size, test_size]
    )

    train_cls = NoBBoxWrapper(train_set)
    val_cls = NoBBoxWrapper(val_set)
    test_cls = NoBBoxWrapper(test_set)

    train_loader = DataLoader(
        train_cls, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_cls, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_cls, batch_size=batch_size, shuffle=False, num_workers=4
    )

    resnet = ResNet18

    if not os.path.exists("models/resnet18.pth"):
        train_classifier(resnet, train_loader, val_loader, train_cls, val_cls)
    else:
        resnet.load_state_dict(torch.load("models/resnet18.pth", map_location=device))

    if not os.path.exists("data/CAM"):
        loader_names = ["Train", "Val", "Test"]
        for loader, name in zip([train_loader, val_loader, test_loader], loader_names):
            for x, y, ids in tqdm(loader, desc=f"[CAM {name}]"):
                create_cam(resnet, x.to(device), y.to(device), ids)

    evaluate_cam_iou(test_loader)
