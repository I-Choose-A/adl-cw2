import os
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm

# 文件路径
img_dir = 'data/images'
trimap_dir = 'data/annotations/trimaps'
output_img_dir = 'data/bbox_images'
os.makedirs(output_img_dir, exist_ok=True)

output_label_file = 'data/annotations/bbox_labels.txt'
f = open(output_label_file, 'w')

resize = transforms.Resize((256, 256))

all_trimap_files = sorted([f for f in os.listdir(trimap_dir) if f.lower().endswith('.png')])

for filename in tqdm(all_trimap_files):
    image_id = os.path.splitext(filename)[0]
    trimap_path = os.path.join(trimap_dir, filename)
    img_path = os.path.join(img_dir, image_id + '.jpg')

    # 如果原图不存在，就跳过
    if not os.path.exists(img_path):
        continue

    # 加载 trimap 和原图
    trimap = Image.open(trimap_path)
    trimap = resize(trimap)
    trimap_np = np.array(trimap)

    # 将 trimap 值映射为前景（1）与背景（0）
    trimap_np[trimap_np == 2] = 0  # 去掉背景
    trimap_np[trimap_np == 3] = 1  # 不确定的也包括
    trimap_np[trimap_np == 1] = 1  # 前景

    # 计算前景坐标
    y_indices, x_indices = np.where(trimap_np == 1)
    if len(x_indices) == 0 or len(y_indices) == 0:
        continue  # 没有前景，跳过

    xmin, xmax = x_indices.min(), x_indices.max()
    ymin, ymax = y_indices.min(), y_indices.max()

    # 写入 bbox 坐标
    f.write(f'{image_id} {xmin} {ymin} {xmax} {ymax}\n')

    # 在图像上画框
    image = Image.open(img_path).convert('RGB')
    image = resize(image)
    draw = ImageDraw.Draw(image)
    draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)

    image.save(os.path.join(output_img_dir, f'{image_id}.jpg'))

f.close()
