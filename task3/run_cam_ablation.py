import os
import shutil
import torch
from train_resnet_cam import train_classifier, evaluate_cam_iou, get_resnet_model, NoBBoxWrapper
from data.dataset_bbox import OxfordIIITPet
from torch.utils.data import DataLoader, random_split
from utils.mask_utils import create_cam, get_trimap, get_cam
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2025)
batch_size = 32
result_log = "cam_ablation_results.txt"

def clear_model_and_cam(model_name):
    # 清空旧模型
    for path in [f"models/{model_name}.pth", f"models/best_{model_name}.pth"]:
        if os.path.exists(path):
            os.remove(path)
    # 清空 CAM
    if os.path.exists("data/CAM"):
        shutil.rmtree("data/CAM")

def prepare_data():
    dataset = OxfordIIITPet()
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_cls = NoBBoxWrapper(train_set)
    val_cls = NoBBoxWrapper(val_set)
    test_cls = NoBBoxWrapper(test_set)

    return (
        DataLoader(train_cls, batch_size=batch_size, shuffle=True),
        DataLoader(val_cls, batch_size=batch_size, shuffle=False),
        DataLoader(test_cls, batch_size=batch_size, shuffle=False),
        train_cls,
        val_cls,
        test_cls,
    )

def train_and_generate_cam(resnet_name):
    print(f"Training and generating CAM for {resnet_name}...")

    clear_model_and_cam(resnet_name)
    train_loader, val_loader, test_loader, train_set, val_set, test_set = prepare_data()

    resnet = get_resnet_model(resnet_name)
    train_classifier(resnet, train_loader, val_loader, train_set, val_set)

    print("Creating CAMs...")
    for loader in [train_loader, val_loader, test_loader]:
        for x, y, ids in tqdm(loader, desc=f"[CAM for {resnet_name}]"):
            create_cam(resnet, x.to(device), y.to(device), ids)

    return test_loader

def run_threshold_experiment(resnet_name, test_loader, thresholds):
    with open(result_log, "a") as f:
        for thresh in thresholds:
            print(f"Evaluating {resnet_name} @ CAM Threshold={thresh}")
            total_iou = 0.
            cam_vals = []

            with torch.no_grad():
                for x, _, image_ids in test_loader:
                    trimap = get_trimap(image_ids).to(device)
                    cam = get_cam(image_ids).to(device)

                    # 统计 CAM 值分布
                    cam_vals.append(cam)

                    cam_bin = (cam > thresh).float()
                    iou = (cam_bin * trimap).sum((1, 2, 3)) / (
                        (cam_bin + trimap).clamp(0, 1).sum((1, 2, 3)) + 1e-6)
                    total_iou += iou.sum().item()

            avg_iou = total_iou / len(test_loader.dataset)
            print(f"[{resnet_name} @ {thresh}] CAM IoU = {avg_iou:.4f}")
            f.write(f"ResNet: {resnet_name}, CAM Thresh: {thresh}, "
                    f"IoU: {avg_iou:.4f}\n")


if __name__ == "__main__":
    # 删除旧结果
    if os.path.exists(result_log):
        os.remove(result_log)

    # ResNet18 + 多个 CAM 阈值
    test_loader = train_and_generate_cam("resnet18")
    run_threshold_experiment("resnet18", test_loader, [0.3, 0.5, 0.7])

    # ResNet34 + 0.5
    test_loader = train_and_generate_cam("resnet34")
    run_threshold_experiment("resnet34", test_loader, [0.3, 0.5, 0.7])

    # ResNet50 + 0.5
    test_loader = train_and_generate_cam("resnet50")
    run_threshold_experiment("resnet50", test_loader, [0.3, 0.5, 0.7])
