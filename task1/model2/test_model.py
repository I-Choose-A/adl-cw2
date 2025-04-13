import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# 添加项目根目录到Python模块搜索路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from Task2_Weakly_Supersive_learning.models.affinitynet import AffinityNet
from Task2_Weakly_Supersive_learning.utils.datasets import PetDataset
from Task2_Weakly_Supersive_learning.utils.transforms import get_transforms
from Task2_Weakly_Supersive_learning.utils.misc import visualize_cam, visualize_affinity

def load_model(model_path, num_classes=37, backbone='resnet50'):
    """加载预训练模型"""
    # 创建模型
    model = AffinityNet(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=False,
        dilated=True
    )
    
    # 加载检查点
    print(f"Loading model: {model_path}")
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 检查检查点内容
    print("Checkpoint keys:", checkpoint.keys())
    
    # 模型状态加载
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Model loaded from epoch {checkpoint['epoch']}, best performance: {checkpoint['best_perf']:.4f}")
    
    # 切换到评估模式
    model.eval()
    
    return model

def inspect_model_weights(model):
    """检查模型权重"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 检查梯度状态（如果训练过，通常会有梯度）
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"Model has gradients: {has_grad}")
    
    # 检查一些关键权重的统计数据
    # 1. 分类头权重
    cls_weights = model.cam_head.weight.data
    print(f"\nClassification head weights statistics:")
    print(f"  Shape: {cls_weights.shape}")
    print(f"  Mean: {cls_weights.mean().item():.6f}")
    print(f"  Std: {cls_weights.std().item():.6f}")
    print(f"  Min: {cls_weights.min().item():.6f}")
    print(f"  Max: {cls_weights.max().item():.6f}")
    
    # 2. 亲和力头权重
    aff_weights = model.affinity_head[-1].weight.data
    print(f"\nAffinity head weights statistics:")
    print(f"  Shape: {aff_weights.shape}")
    print(f"  Mean: {aff_weights.mean().item():.6f}")
    print(f"  Std: {aff_weights.std().item():.6f}")
    print(f"  Min: {aff_weights.min().item():.6f}")
    print(f"  Max: {aff_weights.max().item():.6f}")
    
    # 正常训练的模型，权重应该有分散的值，而不是全部接近于0或者非常相似

def test_forward_pass(model, image_path=None):
    """测试模型前向传播"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 准备输入数据
    if image_path is None or not os.path.exists(image_path):
        # 如果没有提供图像，创建随机输入
        print("Using random generated image for testing")
        x = torch.randn(1, 3, 448, 448).to(device)
    else:
        # 加载和预处理图像
        print(f"Using image {image_path} for testing")
        transform = get_transforms('test')
        img = Image.open(image_path).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(x)
    
    # 检查输出
    print("\nModel outputs:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, value range=[{v.min().item():.4f}, {v.max().item():.4f}]")
    
    # 对分类结果进行排序
    if 'cam_logits' in outputs:
        logits = outputs['cam_logits'].cpu().numpy()[0]
        top_indices = np.argsort(logits)[::-1][:5]  # 获取前5个预测
        
        # 类别名称列表
        classes = [
            'Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound',
            'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer', 'British_Shorthair',
            'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter',
            'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin',
            'keeshond', 'leonberger', 'Maine_Coon', 'miniature_pinscher',
            'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue',
            'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu',
            'Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier',
            'yorkshire_terrier'
        ]
        
        print("\nTop 5 predictions:")
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. {classes[idx]}: {logits[idx]:.4f}")
    
    return outputs, x

def main():
    # 模型路径
    model_path = "D:/Project/adl-cw2/output/stage2_resnet50/model_best.pth.tar"
    
    # 加载模型
    model = load_model(model_path)
    
    # 检查模型权重
    inspect_model_weights(model)
    
    # 测试前向传播
    # 如果有测试图像可以提供路径
    test_image = "data/oxford-iiit-pet/images/Abyssinian_1.jpg"
    if os.path.exists(test_image):
        outputs, x = test_forward_pass(model, test_image)
        
        # 可视化结果
        if torch.cuda.is_available():
            x = x.cpu()
        
        # 可视化CAM
        if 'cam' in outputs:
            cam = outputs['cam'].cpu()
            img = x[0].cpu()
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(img.permute(1, 2, 0).numpy())
            plt.title('Original Image')
            
            plt.subplot(1, 2, 2)
            vis_cam = visualize_cam(img, cam[0])
            plt.imshow(vis_cam)
            plt.title('Class Activation Map (CAM)')
            
            plt.savefig('test_cam_output.png')
            print("Visualization result saved as test_cam_output.png")
    else:
        test_forward_pass(model)
    
    print("\nIf the model shows reasonable parameter distribution and outputs, it means the model has been trained effectively.")
    print("If all weights are close to 0 or very similar, the model may not have been trained correctly.")

if __name__ == "__main__":
    main() 