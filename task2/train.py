import datetime
import os.path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data.dataset import OxfordIIITPet
from eval import eval_classifier
from models.resnet import ResNet18
from models.unet import UNet
from utils.loss import weighted_loss
from utils.mask_utils import create_cam, get_cam, get_trimap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2025)
batch_size = 32

dataset = OxfordIIITPet()
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)
test_size = len(dataset) - train_size - val_size

# In order to make the data in the CAM and batch match,
# the data set is disrupted in advance and the Dataloader is loaded without disrupting the train_loader.
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

resnet = ResNet18
unet = UNet()


def train_classifier(model):
    tqdm.write(f"Start training Classifier...")
    epochs = 10
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    loss_function = torch.nn.CrossEntropyLoss()

    model = model.to(device)

    for epoch in range(epochs):
        model.train()

        train_correct = 0.0
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for x, y, _ in train_bar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(y_pred, 1)
            train_correct += (predicted == y).sum().item()
            train_loss += loss.item() * x.shape[0]
            train_bar.set_postfix({"loss": loss.item()})

        train_acc = train_correct / len(train_dataset)
        train_loss /= len(train_dataset)

        # evaluation on val dataset
        model.eval()
        val_correct = 0.0
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
        for x, y, _ in val_bar:
            x = x.to(device)
            y = y.to(device)

            val_batch_correct, val_batch_loss = eval_classifier(model, x, y)
            val_correct += val_batch_correct
            val_loss += val_batch_loss * x.shape[0]
            val_bar.set_postfix({"loss": val_batch_loss})

        val_acc = val_correct / len(val_dataset)
        val_loss /= len(val_dataset)

        train_bar.clear()
        val_bar.clear()
        tqdm.write(
            f"EPOCH: {epoch + 1}/{epochs}, train_loss: {train_loss}, train_acc: {train_acc * 100:.2f}% "
            f"val_loss: {val_loss}, val_acc: {val_acc * 100:.2f}%"
        )

    torch.save(model.state_dict(), "models/resnet18.pth")


def train_unet(model):
    tqdm.write(f"Start training UNet...")

    epochs = 10
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5, weight_decay=1e-6)
    model = model.to(device)

    for epoch in range(epochs):
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for x, _, image_ids in train_bar:
            x = x.to(device)

            optimizer.zero_grad()
            cam = get_cam(image_ids)

            mask = cam

            pred_mask = model(x)
            loss = weighted_loss(pred_mask, mask)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.shape[0]
            train_bar.set_postfix({"loss": loss.item()})
        train_loss /= len(train_dataset)

        # evaluation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
        for x, _, image_ids in val_bar:
            x = x.to(device)
            trimap = get_trimap(image_ids).to(device)

            with torch.no_grad():
                pred_mask = model(x)
                loss = weighted_loss(pred_mask, trimap)
                val_loss += loss.item() * x.shape[0]

                pred_binary = (pred_mask > 0.5).float()
                intersection = (pred_binary * trimap).sum((1, 2, 3))
                union = (pred_binary + trimap).clamp(0, 1).sum((1, 2, 3))
                batch_iou = (intersection / (union + 1e-6)).sum().item()
                val_iou += batch_iou

                val_bar.set_postfix({"loss": loss.item()})

        val_loss /= len(val_dataset)
        val_iou /= len(val_dataset)

        train_bar.clear()
        val_bar.clear()
        tqdm.write(
            f"EPOCH: {epoch + 1}/{epochs}, train_loss: {train_loss}, val_loss: {val_loss}, val_iou:{val_iou}"
        )

    torch.save(model.state_dict(), "models/unet.pth")


#
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.clone()
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    tensor.mul_(std).add_(mean)  # x = (x_norm * std) + mean
    return tensor.clamp_(0, 1)  # clip to [0,1]


if __name__ == "__main__":
    if not os.path.exists("models/resnet18.pth"):
        train_classifier(resnet)
    else:
        resnet.load_state_dict(torch.load("models/resnet18.pth"))

    if not os.path.exists("data/CAM"):
        for loader in [train_loader, val_loader, test_loader]:
            # for loader in [test_loader]:
            cam_bar = tqdm(loader, desc=f"创建CAM")
            for i, (x, y, ids) in enumerate(cam_bar):
                create_cam(resnet, x, y, ids)
                cam_bar.set_postfix({"batch": i})

    # clear cache to provide more space for training UNet
    resnet = resnet.to("cpu")
    torch.cuda.empty_cache()

    if not os.path.exists("models/unet.pth"):
        train_unet(unet)
    else:
        unet.load_state_dict(torch.load("models/unet.pth"))

    # test
    unet.eval()
    unet = unet.to(device)

    test_loss = 0.0
    test_iou = 0.0
    cam_iou = 0.0
    for x, _, image_ids in test_loader:
        x = x.to(device)
        trimap = get_trimap(image_ids).to(device)

        with torch.no_grad():
            pred_mask = unet(x)
            loss = weighted_loss(pred_mask, trimap)
            test_loss += loss.item()

            pred_binary = (pred_mask > 0.5).float()
            intersection = (pred_binary * trimap).sum((1, 2, 3))
            union = (pred_binary + trimap).clamp(0, 1).sum((1, 2, 3))
            batch_iou = (intersection / (union + 1e-6)).sum().item()
            test_iou += batch_iou

            cam = get_cam(image_ids)
            cam_binary = (cam > 0.5).float()
            cam_intersection = (cam_binary * trimap).sum((1, 2, 3))
            cam_union = (cam_binary + trimap).clamp(0, 1).sum((1, 2, 3))
            cam_batch_iou = (cam_intersection / (cam_union + 1e-6)).sum().item()
            cam_iou += cam_batch_iou

    test_loss /= len(test_dataset)
    test_iou /= len(test_dataset)
    cam_iou /= len(test_dataset)

    print(f"test_loss: {test_loss},test_iou:{test_iou}, {datetime.datetime.now()}, cam_iou:{cam_iou}")

    # display samples
    unet.eval()
    # creating directories to save images
    os.makedirs("output_images/train", exist_ok=True)
    os.makedirs("output_images/val", exist_ok=True)
    os.makedirs("output_images/test", exist_ok=True)


    def save_images(loader, folder_name):
        print(f"Saving {folder_name} images...")
        for i, (x, y, image_ids) in enumerate(
                tqdm(loader, desc=f"Saving {folder_name} images")
        ):
            x = x.to(device)

            # process each image in current batch
            for j in range(len(image_ids)):
                # create a wide canvas to hold all images
                combined_width = 224 * 4  # Assuming image width is 224
                combined_height = 224 + 30  # Extra height for titles
                combined_image = Image.new(
                    "RGB", (combined_width, combined_height), (255, 255, 255)
                )

                # original image
                x_denorm = denormalize(x[j].unsqueeze(0).to("cpu"))
                image_np = x_denorm.squeeze(0).permute(1, 2, 0).numpy()
                image_uint8 = (image_np * 255).astype(np.uint8)
                original_img = Image.fromarray(image_uint8)

                # trimap
                trimap = get_trimap([image_ids[j]])
                trimap_binary = (trimap[0].squeeze(0) > 0.5).float() * 255
                trimap_img = Image.fromarray(
                    trimap_binary.to("cpu").numpy().astype(np.uint8), mode="L"
                ).convert("RGB")

                # predicted mask
                with torch.no_grad():
                    pred_mask = unet(x[j].unsqueeze(0))

                mask_binary = (pred_mask[0].squeeze(0) > 0.5).float() * 255
                mask_img = Image.fromarray(
                    mask_binary.to("cpu").numpy().astype(np.uint8), mode="L"
                ).convert("RGB")

                # CAM
                cam = get_cam([image_ids[j]])
                cam_binary = cam[0].squeeze(0) * 255
                cam_img = Image.fromarray(
                    cam_binary.to("cpu").numpy().astype(np.uint8), mode="L"
                ).convert("RGB")

                # composite images
                combined_image.paste(original_img, (0, 30))
                combined_image.paste(trimap_img, (224, 30))
                combined_image.paste(mask_img, (224 * 2, 30))
                combined_image.paste(cam_img, (224 * 3, 30))

                # add titles
                draw = ImageDraw.Draw(combined_image)
                try:
                    # Try loading font, fallback to default if fails
                    font = ImageFont.truetype("Arial", 18)
                except IOError:
                    font = ImageFont.load_default()

                # draw subtitles
                titles = ["Original", "Trimap", "Predicted Mask", "CAM"]
                for idx, title in enumerate(titles):
                    # Calculate text position for center alignment
                    text_width = font.getbbox(title)[2] - font.getbbox(title)[0]
                    position = (idx * 224 + (224 - text_width) // 2, 5)
                    draw.text(position, title, fill=(0, 0, 0), font=font)

                # save composited image
                combined_image.save(
                    f"output_images/{folder_name}/combined_{image_ids[j]}.png"
                )


    # save images for train, validation and test sets
    save_images(train_loader, "train")
    save_images(val_loader, "val")
    save_images(test_loader, "test")

    print("All images saved to train, val and test subdirectories under output_images")
