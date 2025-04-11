import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import kornia.filters as kf  # 添加kornia用于边缘检测

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 内存优化的PyTorch CRF实现 - 修复版
class DenseCRF(torch.nn.Module):
    def __init__(self, iter_max=5):
        super(DenseCRF, self).__init__()
        self.iter_max = iter_max

    def forward(
        self,
        img,
        unary,
        sxy_gaussian=3,
        compat_gaussian=10,
        sxy_bilateral=20,
        srgb_bilateral=10,
        compat_bilateral=3,
    ):
        """
        Args:
            img: RGB图像 [3, H, W]
            unary: 一元势 [2, H, W] - 背景和前景的概率
            其他参数：CRF的超参数
        """
        h, w = img.shape[1], img.shape[2]
        n_labels = unary.shape[0]

        # 使用更小的图像尺寸进行处理，避免内存问题
        max_side = 64
        if h > max_side or w > max_side:
            scale_factor = max_side / max(h, w)
            img_small = F.interpolate(
                img.unsqueeze(0),
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=False,
            )[0]
            unary_small = F.interpolate(
                unary.unsqueeze(0),
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=False,
            )[0]

            # 直接处理小图像版本
            Q_small = self._dense_crf(
                img_small,
                unary_small,
                sxy_gaussian,
                compat_gaussian,
                sxy_bilateral,
                srgb_bilateral,
                compat_bilateral,
            )

            # 上采样回原尺寸
            Q = F.interpolate(
                Q_small.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            )[0]
            return Q
        else:
            return self._dense_crf(
                img,
                unary,
                sxy_gaussian,
                compat_gaussian,
                sxy_bilateral,
                srgb_bilateral,
                compat_bilateral,
            )

    def _dense_crf(
        self,
        img,
        unary,
        sxy_gaussian=3,
        compat_gaussian=10,
        sxy_bilateral=20,
        srgb_bilateral=10,
        compat_bilateral=3,
    ):
        """实际的CRF计算，用于较小图像"""
        h, w = img.shape[1], img.shape[2]
        n_labels = unary.shape[0]
        n_pixels = h * w

        # 初始化Q为一元势的概率
        Q = unary.clone()

        # 直接使用高斯滤波器方法，避免全尺寸成对矩阵
        for _ in range(self.iter_max):
            # 对Q进行softmax处理
            Q = F.softmax(Q, dim=0)

            # 准备消息
            message = torch.zeros_like(Q)

            # 1. 空间高斯消息传递 - 使用可分离卷积近似
            for label in range(n_labels):
                # 使用高斯模糊近似消息传递
                feat = Q[label : label + 1]  # [1, H, W]
                # 滤波器大小与sigma成正比
                ksize = int(2 * sxy_gaussian) * 2 + 1
                # 对特征图应用可分离高斯滤波
                blurred = self._gaussian_blur(feat, ksize, sxy_gaussian)
                # 计算消息
                message[label : label + 1] += compat_gaussian * (blurred - feat)

            # 2. 双边消息传递 - 使用引导滤波器近似
            for label in range(n_labels):
                # 使用引导滤波近似双边滤波
                feat = Q[label : label + 1]  # [1, H, W]
                guided = self._guided_filter(img, feat, sxy_bilateral, srgb_bilateral)
                message[label : label + 1] += compat_bilateral * (guided - feat)

            # 更新Q
            Q = unary - message

        # 最终的Q
        Q = F.softmax(Q, dim=0)
        return Q

    def _gaussian_blur(self, x, kernel_size, sigma):
        """使用可分离卷积实现高斯模糊"""
        # 创建1D高斯核
        grid = (
            torch.arange(kernel_size, device=x.device).float() - (kernel_size - 1) / 2
        )
        gaussian = torch.exp(-(grid**2) / (2 * sigma**2))
        kernel = gaussian / gaussian.sum()

        # 应用可分离卷积
        padding = (kernel_size - 1) // 2

        # 创建卷积权重
        weight_h = kernel.view(1, 1, kernel_size, 1).repeat(1, 1, 1, 1)
        weight_w = kernel.view(1, 1, 1, kernel_size).repeat(1, 1, 1, 1)

        # 水平方向卷积
        x = F.conv2d(x, weight_h, padding=(padding, 0), groups=1)
        # 垂直方向卷积
        x = F.conv2d(x, weight_w, padding=(0, padding), groups=1)
        return x

    def _guided_filter(self, guide, src, radius, eps):
        """引导滤波器，用于近似双边滤波"""
        guide = guide.mean(dim=0, keepdim=True)  # [1, H, W]

        # 均值滤波器
        kernel_size = int(2 * radius) + 1
        padding = (kernel_size - 1) // 2

        # 创建均值卷积核
        mean_kernel = torch.ones(
            1, 1, kernel_size, kernel_size, device=guide.device
        ) / (kernel_size**2)

        # 计算均值
        mean_guide = F.conv2d(guide, mean_kernel, padding=padding)
        mean_src = F.conv2d(src, mean_kernel, padding=padding)
        mean_guide_src = F.conv2d(guide * src, mean_kernel, padding=padding)

        # 计算协方差
        cov_guide_src = mean_guide_src - mean_guide * mean_src

        # 计算自方差
        var_guide = (
            F.conv2d(guide * guide, mean_kernel, padding=padding) - mean_guide**2
        )

        # 计算a和b
        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide

        # 计算均值a和均值b
        mean_a = F.conv2d(a, mean_kernel, padding=padding)
        mean_b = F.conv2d(b, mean_kernel, padding=padding)

        # 最终输出
        output = mean_a * guide + mean_b
        return output


# save grad_cam in local as .pt file
def create_cam(model, x, y, image_ids):
    os.makedirs("data/CAM", exist_ok=True)
    model = model.to(device)
    model.eval()
    x = x.to(device)
    y = y.to(device)

    # 数据增强生成多个视角
    transforms_list = [
        # 原始图像
        lambda img: img,
        # 水平翻转
        lambda img: torch.flip(img, dims=[-1]),
        # 旋转10度
        lambda img: torch.nn.functional.affine_grid(
            torch.tensor([[[1, 0, 0], [0, 1, 0]]], device=device)
            * torch.cos(torch.tensor(10 * np.pi / 180)),
            size=torch.Size((1, img.shape[0], img.shape[1], img.shape[2])),
            align_corners=False,
        ).to(device),
        # 缩放0.9倍
        lambda img: F.interpolate(
            img.unsqueeze(0), scale_factor=0.9, mode="bilinear", align_corners=False
        )[0],
    ]

    # 存储所有增强版本的CAM
    all_cams = []

    # 对每个增强版本生成CAM
    for transform_fn in transforms_list:
        # 跳过需要复杂操作的变换（仅保留前两种简单变换用于演示）
        if transform_fn == transforms_list[2] or transform_fn == transforms_list[3]:
            continue

        # 应用变换
        transformed_x = torch.stack([transform_fn(img) for img in x])

        # 收集不同层级的特征图
        features_layer3 = []
        features_layer4 = []
        gradients_layer3 = []
        gradients_layer4 = []

        # hook方法收集特征图
        def hook_feature_layer3(module, input, output):
            features_layer3.append(output)

        def hook_feature_layer4(module, input, output):
            features_layer4.append(output)

        # hook方法收集梯度图
        def hook_grad_layer3(module, grad_input, grad_output):
            gradients_layer3.append(grad_output[0])

        def hook_grad_layer4(module, grad_input, grad_output):
            gradients_layer4.append(grad_output[0])

        # 注册前向和反向钩子
        handle_layer3 = model.layer3[-1].conv2.register_forward_hook(
            hook_feature_layer3
        )
        handle_layer4 = model.layer4[-1].conv2.register_forward_hook(
            hook_feature_layer4
        )
        handle_grad_layer3 = model.layer3[-1].conv2.register_full_backward_hook(
            hook_grad_layer3
        )
        handle_grad_layer4 = model.layer4[-1].conv2.register_full_backward_hook(
            hook_grad_layer4
        )

        # 前向传播
        logits = model(transformed_x)

        model.zero_grad()  # 梯度清零

        # 只计算目标类别的梯度
        one_hot_y = F.one_hot(y, num_classes=37).to(device)
        logits.backward(gradient=one_hot_y, retain_graph=True)

        # 移除hook
        handle_layer3.remove()
        handle_layer4.remove()
        handle_grad_layer3.remove()
        handle_grad_layer4.remove()

        # 只使用第一次调用的数据
        features_layer3 = features_layer3[0]  # shape = (B,C,H,W)
        features_layer4 = features_layer4[0]  # shape = (B,C,H,W)
        gradients_layer3 = gradients_layer3[0]  # shape = (B,C,H,W)
        gradients_layer4 = gradients_layer4[0]  # shape = (B,C,H,W)

        # 分别计算两个层的GradCAM++
        # Layer 3 GradCAM++
        grad_2_layer3 = gradients_layer3**2
        grad_3_layer3 = grad_2_layer3 * gradients_layer3

        alpha_num_layer3 = grad_2_layer3
        alpha_denom_layer3 = 2 * grad_2_layer3 + features_layer3 * grad_3_layer3 + 1e-8

        alpha_layer3 = alpha_num_layer3 / alpha_denom_layer3
        alpha_norm_layer3 = alpha_layer3 / (
            torch.sum(alpha_layer3, dim=[1, 2, 3], keepdim=True) + 1e-8
        )

        cam_layer3 = torch.sum(alpha_norm_layer3 * F.relu(features_layer3), dim=1)
        cam_layer3 = F.relu(cam_layer3)

        # Layer 4 GradCAM++
        grad_2_layer4 = gradients_layer4**2
        grad_3_layer4 = grad_2_layer4 * gradients_layer4

        alpha_num_layer4 = grad_2_layer4
        alpha_denom_layer4 = 2 * grad_2_layer4 + features_layer4 * grad_3_layer4 + 1e-8

        alpha_layer4 = alpha_num_layer4 / alpha_denom_layer4
        alpha_norm_layer4 = alpha_layer4 / (
            torch.sum(alpha_layer4, dim=[1, 2, 3], keepdim=True) + 1e-8
        )

        cam_layer4 = torch.sum(alpha_norm_layer4 * F.relu(features_layer4), dim=1)
        cam_layer4 = F.relu(cam_layer4)

        # 将两层的CAM调整到相同大小
        cam_layer3 = F.interpolate(
            cam_layer3.unsqueeze(1),
            size=cam_layer4.shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # 归一化
        cam_layer3 = (cam_layer3 - cam_layer3.amin(dim=(1, 2), keepdim=True)[0]) / (
            cam_layer3.amax(dim=(1, 2), keepdim=True)[0] + 1e-8
        )

        cam_layer4 = (cam_layer4 - cam_layer4.amin(dim=(1, 2), keepdim=True)[0]) / (
            cam_layer4.amax(dim=(1, 2), keepdim=True)[0] + 1e-8
        )

        # 多尺度融合 - 高层特征权重更大，低层特征权重更小
        cam = 0.7 * cam_layer4 + 0.3 * cam_layer3

        # 再次归一化
        cam = (cam - cam.amin(dim=(1, 2), keepdim=True)[0]) / (
            cam.amax(dim=(1, 2), keepdim=True)[0] + 1e-8
        )

        # 上采样到输入尺寸
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # 如果是水平翻转的数据，需要翻转回来
        if transform_fn == transforms_list[1]:
            cam = torch.flip(cam, dims=[-1])

        # 添加到列表
        all_cams.append(cam)

    # 集成多个视角的CAM (取平均)
    ensemble_cam = torch.mean(torch.cat(all_cams, dim=1), dim=1, keepdim=True)

    # 边缘感知增强
    for i in range(x.shape[0]):
        # 获取原始CAM和图像
        img_tensor = x[i]  # [3, H, W]
        cam_tensor = ensemble_cam[i, 0]  # [H, W]

        # 使用Sobel算子提取边缘
        edges = kf.sobel(img_tensor.mean(dim=0, keepdim=True).unsqueeze(0))
        edges = F.interpolate(
            edges,
            size=(cam_tensor.shape[0], cam_tensor.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        edges = edges.squeeze()  # [H, W]

        # 归一化边缘强度
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)

        # 边缘区域CAM值增强
        edge_weight = edges > 0.2  # 边缘阈值筛选

        # 边缘感知调整：锐化边界处的CAM值
        cam_tensor_adjusted = cam_tensor.clone()
        # 提高边缘处的CAM对比度
        cam_tensor_adjusted = torch.where(
            edge_weight,
            torch.clamp(cam_tensor * 1.2, 0, 1),  # 边缘处增强
            cam_tensor,  # 非边缘处保持不变
        )

        # 更新CAM
        ensemble_cam[i, 0] = cam_tensor_adjusted

        # 保存结果
        single_cam = ensemble_cam[i, 0].detach().cpu()  # (H,W)
        torch.save(single_cam, f"data/CAM/{image_ids[i]}.pt")

    # 创建CRF模型并移至GPU
    dense_crf = DenseCRF(iter_max=15).to(device)  # 增加迭代次数

    # CRF Processing (per image in batch)
    for i in range(x.shape[0]):
        # 获取图像和CAM数据，保持在GPU上
        img_tensor = x[i].to(device)  # [3, H, W]
        cam_tensor = ensemble_cam[i, 0].to(device)  # [H, W]

        # 清理GPU缓存减少内存碎片
        torch.cuda.empty_cache()

        # 跳过无效图像
        if torch.max(cam_tensor) < 0.1:
            continue

        # 准备一元势（在GPU上）
        unary = torch.stack([1 - cam_tensor, cam_tensor], dim=0)  # [2, H, W]

        # 执行CRF推理
        Q = dense_crf(
            img_tensor,
            unary,
            sxy_gaussian=4,  # 调整空间高斯参数
            compat_gaussian=12,  # 调整兼容性参数
            sxy_bilateral=25,  # 调整双边滤波空间参数
            srgb_bilateral=15,  # 调整双边滤波颜色参数
            compat_bilateral=5,
        )  # 调整双边滤波兼容性参数

        # 获取最终结果
        refined = torch.argmax(Q, dim=0)  # [H, W]

        # 保存结果
        torch.save(refined, f"data/CAM/{image_ids[i]}.pt")


def get_cam(image_ids):
    cam = [
        torch.load(f"data/CAM/{image_id}.pt").reshape(1, 1, 256, 256).to(device)
        for image_id in image_ids
    ]
    return torch.cat(cam, dim=0)


def get_trimap(image_ids):
    trimaps = []
    for image_id in image_ids:
        trimap_path = f"data/annotations/trimaps/{image_id}.png"
        trimap = Image.open(trimap_path)
        trimap = transforms.Resize((256, 256))(trimap)

        trimap = np.array(trimap)
        trimap[trimap == 2] = 0
        trimap[trimap == 3] = 1
        trimap = torch.from_numpy(trimap).float().reshape(1, 1, 256, 256)

        trimaps.append(trimap)
    return torch.cat(trimaps, dim=0)
