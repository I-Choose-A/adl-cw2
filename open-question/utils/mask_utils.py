import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import kornia.filters as kf  # Add kornia for edge detection

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Memory-optimized PyTorch CRF implementation - Fixed version
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
            img: RGB image [3, H, W]
            unary: Unary potentials [2, H, W] - probabilities for background and foreground
            Other parameters: CRF hyperparameters
        """
        h, w = img.shape[1], img.shape[2]
        n_labels = unary.shape[0]

        # Use smaller image size for processing to avoid memory issues
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

            # Process smaller image version directly
            Q_small = self._dense_crf(
                img_small,
                unary_small,
                sxy_gaussian,
                compat_gaussian,
                sxy_bilateral,
                srgb_bilateral,
                compat_bilateral,
            )

            # Upsample back to original size
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
        """Actual CRF computation for smaller images"""
        h, w = img.shape[1], img.shape[2]
        n_labels = unary.shape[0]
        n_pixels = h * w

        # Initialize Q with unary potentials probability
        Q = unary.clone()

        # Use Gaussian filter method directly to avoid full-size pairwise matrices
        for _ in range(self.iter_max):
            # Apply softmax to Q
            Q = F.softmax(Q, dim=0)

            # Prepare messages
            message = torch.zeros_like(Q)

            # 1. Spatial Gaussian message passing - using separable convolution approximation
            for label in range(n_labels):
                # Use Gaussian blur to approximate message passing
                feat = Q[label : label + 1]  # [1, H, W]
                # Filter size proportional to sigma
                ksize = int(2 * sxy_gaussian) * 2 + 1
                # Apply separable Gaussian filter to feature map
                blurred = self._gaussian_blur(feat, ksize, sxy_gaussian)
                # Calculate message
                message[label : label + 1] += compat_gaussian * (blurred - feat)

            # 2. Bilateral message passing - using guided filter approximation
            for label in range(n_labels):
                # Use guided filter to approximate bilateral filter
                feat = Q[label : label + 1]  # [1, H, W]
                guided = self._guided_filter(img, feat, sxy_bilateral, srgb_bilateral)
                message[label : label + 1] += compat_bilateral * (guided - feat)

            # Update Q
            Q = unary - message

        # Final Q
        Q = F.softmax(Q, dim=0)
        return Q

    def _gaussian_blur(self, x, kernel_size, sigma):
        """Use separable convolution to implement Gaussian blur"""
        # Create 1D Gaussian kernel
        grid = (
            torch.arange(kernel_size, device=x.device).float() - (kernel_size - 1) / 2
        )
        gaussian = torch.exp(-(grid**2) / (2 * sigma**2))
        kernel = gaussian / gaussian.sum()

        # Apply separable convolution
        padding = (kernel_size - 1) // 2

        # Create convolution weights
        weight_h = kernel.view(1, 1, kernel_size, 1).repeat(1, 1, 1, 1)
        weight_w = kernel.view(1, 1, 1, kernel_size).repeat(1, 1, 1, 1)

        # Horizontal direction convolution
        x = F.conv2d(x, weight_h, padding=(padding, 0), groups=1)
        # Vertical direction convolution
        x = F.conv2d(x, weight_w, padding=(0, padding), groups=1)
        return x

    def _guided_filter(self, guide, src, radius, eps):
        """Guided filter for approximate bilateral filter"""
        # Record original dimensions for later processing
        guide_dim = guide.dim()
        src_dim = src.dim()

        # Ensure guide and src are 4D tensors [N,C,H,W]
        if guide_dim == 3:  # [C,H,W]
            guide = guide.unsqueeze(0)
        elif guide_dim == 2:  # [H,W]
            guide = guide.unsqueeze(0).unsqueeze(0)

        if src_dim == 3:  # [C,H,W]
            src = src.unsqueeze(0)
        elif src_dim == 2:  # [H,W]
            src = src.unsqueeze(0).unsqueeze(0)

        # Ensure guide and src have the same spatial dimensions
        if guide.shape[2:] != src.shape[2:]:
            src = F.interpolate(
                src,
                size=(guide.shape[2], guide.shape[3]),
                mode="bilinear",
                align_corners=False,
            )

        # Convert multi-channel guide to single channel
        if guide.shape[1] > 1:
            guide = guide.mean(dim=1, keepdim=True)  # [N, 1, H, W]

        # Mean filter
        kernel_size = int(2 * radius) + 1
        padding = (kernel_size - 1) // 2

        # Create mean convolution kernel
        mean_kernel = torch.ones(
            1, 1, kernel_size, kernel_size, device=guide.device
        ) / (kernel_size**2)

        # Calculate mean
        mean_guide = F.conv2d(guide, mean_kernel, padding=padding)
        mean_src = F.conv2d(src, mean_kernel, padding=padding)
        mean_guide_src = F.conv2d(guide * src, mean_kernel, padding=padding)

        # Calculate covariance
        cov_guide_src = mean_guide_src - mean_guide * mean_src

        # Calculate variance
        var_guide = (
            F.conv2d(guide * guide, mean_kernel, padding=padding) - mean_guide**2
        )

        # Calculate a and b
        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide

        # Calculate mean a and mean b
        mean_a = F.conv2d(a, mean_kernel, padding=padding)
        mean_b = F.conv2d(b, mean_kernel, padding=padding)

        # Explicitly check and handle dimensions
        target_size = (
            guide.shape[2],
            guide.shape[3],
        )  # Explicitly specify as (H,W) tuple

        # Ensure dimensions are consistent
        if mean_a.shape[2:] != guide.shape[2:]:
            mean_a = F.interpolate(
                mean_a, size=target_size, mode="bilinear", align_corners=False
            )

        if mean_b.shape[2:] != guide.shape[2:]:
            mean_b = F.interpolate(
                mean_b, size=target_size, mode="bilinear", align_corners=False
            )

        # Final output
        output = mean_a * guide + mean_b

        # Restore original dimensions
        if guide_dim == 3 and output.dim() == 4:
            output = output.squeeze(0)
        elif guide_dim == 2 and output.dim() == 4:
            output = output.squeeze(0).squeeze(0)

        return output


# save grad_cam in local as .pt file
def create_cam(model, x, y, image_ids):
    os.makedirs("data/CAM", exist_ok=True)
    model = model.to(device)
    model.eval()
    x = x.to(device)
    y = y.to(device)

    # Data augmentation generates multiple views
    transforms_list = [
        # Original image
        lambda img: img,
        # Horizontal flip
        lambda img: torch.flip(img, dims=[-1]),
        # Rotate 10 degrees
        lambda img: torch.nn.functional.affine_grid(
            torch.tensor([[[1, 0, 0], [0, 1, 0]]], device=device)
            * torch.cos(torch.tensor(10 * np.pi / 180)),
            size=torch.Size((1, img.shape[0], img.shape[1], img.shape[2])),
            align_corners=False,
        ).to(device),
        # Scale 0.9x
        lambda img: F.interpolate(
            img.unsqueeze(0), scale_factor=0.9, mode="bilinear", align_corners=False
        )[0],
    ]

    # Store all enhanced version CAMs
    all_cams = []

    # Generate CAM for each enhanced version
    for transform_fn in transforms_list:
        # Skip transformations that require complex operations (only retain the first two simple transformations for demonstration)
        if transform_fn == transforms_list[2] or transform_fn == transforms_list[3]:
            continue

        # Apply transformation
        transformed_x = torch.stack([transform_fn(img) for img in x])

        # Collect feature maps from different layers
        features_layer1 = []
        features_layer2 = []
        features_layer3 = []
        features_layer4 = []
        gradients_layer1 = []
        gradients_layer2 = []
        gradients_layer3 = []
        gradients_layer4 = []

        # hook methods to collect feature maps
        def hook_feature_layer1(module, input, output):
            features_layer1.append(output)

        def hook_feature_layer2(module, input, output):
            features_layer2.append(output)

        def hook_feature_layer3(module, input, output):
            features_layer3.append(output)

        def hook_feature_layer4(module, input, output):
            features_layer4.append(output)

        # hook methods to collect gradient maps
        def hook_grad_layer1(module, grad_input, grad_output):
            gradients_layer1.append(grad_output[0])

        def hook_grad_layer2(module, grad_input, grad_output):
            gradients_layer2.append(grad_output[0])

        def hook_grad_layer3(module, grad_input, grad_output):
            gradients_layer3.append(grad_output[0])

        def hook_grad_layer4(module, grad_input, grad_output):
            gradients_layer4.append(grad_output[0])

        # Register forward and backward hooks
        handle_layer1 = model.layer1[-1].conv2.register_forward_hook(
            hook_feature_layer1
        )
        handle_layer2 = model.layer2[-1].conv2.register_forward_hook(
            hook_feature_layer2
        )
        handle_layer3 = model.layer3[-1].conv2.register_forward_hook(
            hook_feature_layer3
        )
        handle_layer4 = model.layer4[-1].conv2.register_forward_hook(
            hook_feature_layer4
        )
        handle_grad_layer1 = model.layer1[-1].conv2.register_full_backward_hook(
            hook_grad_layer1
        )
        handle_grad_layer2 = model.layer2[-1].conv2.register_full_backward_hook(
            hook_grad_layer2
        )
        handle_grad_layer3 = model.layer3[-1].conv2.register_full_backward_hook(
            hook_grad_layer3
        )
        handle_grad_layer4 = model.layer4[-1].conv2.register_full_backward_hook(
            hook_grad_layer4
        )

        # Forward propagation
        logits = model(transformed_x)

        model.zero_grad()  # Zero gradients

        # Calculate gradients only for the target class
        one_hot_y = F.one_hot(y, num_classes=37).to(device)
        logits.backward(gradient=one_hot_y, retain_graph=True)

        # Remove hooks
        handle_layer1.remove()
        handle_layer2.remove()
        handle_layer3.remove()
        handle_layer4.remove()
        handle_grad_layer1.remove()
        handle_grad_layer2.remove()
        handle_grad_layer3.remove()
        handle_grad_layer4.remove()

        # Use data from the first call only
        features_layer1 = features_layer1[0]  # shape = (B,C,H,W)
        features_layer2 = features_layer2[0]  # shape = (B,C,H,W)
        features_layer3 = features_layer3[0]  # shape = (B,C,H,W)
        features_layer4 = features_layer4[0]  # shape = (B,C,H,W)
        gradients_layer1 = gradients_layer1[0]  # shape = (B,C,H,W)
        gradients_layer2 = gradients_layer2[0]  # shape = (B,C,H,W)
        gradients_layer3 = gradients_layer3[0]  # shape = (B,C,H,W)
        gradients_layer4 = gradients_layer4[0]  # shape = (B,C,H,W)

        # Add channel attention mechanism
        def channel_attention(features, gradients):
            # Calculate channel importance
            channel_weights = torch.sum(torch.abs(gradients), dim=[2, 3], keepdim=True)
            channel_weights = F.softmax(channel_weights, dim=1)
            # Apply channel attention
            weighted_features = features * channel_weights
            return weighted_features

        # Apply channel attention to all layers
        features_layer1 = channel_attention(features_layer1, gradients_layer1)
        features_layer2 = channel_attention(features_layer2, gradients_layer2)
        features_layer3 = channel_attention(features_layer3, gradients_layer3)
        features_layer4 = channel_attention(features_layer4, gradients_layer4)

        # Calculate GradCAM++ for each layer
        # Define a function to calculate GradCAM++
        def calculate_gradcam_pp(features, gradients):
            grad_2 = gradients**2
            grad_3 = grad_2 * gradients

            alpha_num = grad_2
            alpha_denom = 2 * grad_2 + features * grad_3 + 1e-8

            alpha = alpha_num / alpha_denom
            alpha_norm = alpha / (torch.sum(alpha, dim=[1, 2, 3], keepdim=True) + 1e-8)

            cam = torch.sum(alpha_norm * F.relu(features), dim=1)
            cam = F.relu(cam)
            return cam

        # Calculate CAM for each layer
        cam_layer1 = calculate_gradcam_pp(features_layer1, gradients_layer1)
        cam_layer2 = calculate_gradcam_pp(features_layer2, gradients_layer2)
        cam_layer3 = calculate_gradcam_pp(features_layer3, gradients_layer3)
        cam_layer4 = calculate_gradcam_pp(features_layer4, gradients_layer4)

        # Adjust all layers CAM to the same size (use layer4's size as reference)
        target_size = cam_layer4.shape[1:]

        # Upsample other layers CAM to target_size
        def upsample_cam(cam, target_size):
            return F.interpolate(
                cam.unsqueeze(1), size=target_size, mode="bilinear", align_corners=False
            ).squeeze(1)

        cam_layer1 = upsample_cam(cam_layer1, target_size)
        cam_layer2 = upsample_cam(cam_layer2, target_size)
        cam_layer3 = upsample_cam(cam_layer3, target_size)

        # Normalize all layers CAM
        def normalize_cam(cam):
            return (cam - cam.amin(dim=(1, 2), keepdim=True)[0]) / (
                cam.amax(dim=(1, 2), keepdim=True)[0] + 1e-8
            )

        cam_layer1 = normalize_cam(cam_layer1)
        cam_layer2 = normalize_cam(cam_layer2)
        cam_layer3 = normalize_cam(cam_layer3)
        cam_layer4 = normalize_cam(cam_layer4)

        # Calculate layer clarity metric
        def calculate_clarity(cam):
            # Use gradient magnitude as clarity metric
            grad_x = torch.abs(cam[:, :, 1:] - cam[:, :, :-1])
            grad_y = torch.abs(cam[:, 1:, :] - cam[:, :-1, :])

            # Average gradient magnitude
            clarity = (torch.mean(grad_x) + torch.mean(grad_y)) / 2
            return clarity

        clarity_layer1 = calculate_clarity(cam_layer1)
        clarity_layer2 = calculate_clarity(cam_layer2)
        clarity_layer3 = calculate_clarity(cam_layer3)
        clarity_layer4 = calculate_clarity(cam_layer4)

        # Calculate adaptive weights
        total_clarity = (
            clarity_layer1 + clarity_layer2 + clarity_layer3 + clarity_layer4 + 1e-8
        )

        # Base weights - biased toward higher-level features
        base_weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device)

        # Clarity weights
        clarity_weights = (
            torch.tensor(
                [clarity_layer1, clarity_layer2, clarity_layer3, clarity_layer4],
                device=device,
            )
            / total_clarity
        )

        # Merge base and clarity weights
        alpha = 0.7  # Controls the influence of base weights
        fusion_weights = alpha * base_weights + (1 - alpha) * clarity_weights

        # Normalize weight sum
        fusion_weights = fusion_weights / fusion_weights.sum()

        # Apply multi-layer fusion
        cam = (
            fusion_weights[0] * cam_layer1
            + fusion_weights[1] * cam_layer2
            + fusion_weights[2] * cam_layer3
            + fusion_weights[3] * cam_layer4
        )

        # Normalize again and continue previous processing
        cam = normalize_cam(cam)

        # Upsample to input size
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # If data is horizontally flipped, flip it back
        if transform_fn == transforms_list[1]:
            cam = torch.flip(cam, dims=[-1])

        # Add to list
        all_cams.append(cam)

    # Ensemble CAMs from multiple views (take average)
    ensemble_cam = torch.mean(torch.cat(all_cams, dim=1), dim=1, keepdim=True)

    # Improved edge-aware enhancement
    for i in range(x.shape[0]):
        # Get original CAM and image
        img_tensor = x[i]  # [3, H, W]
        cam_tensor = ensemble_cam[i, 0]  # [H, W]

        # Multi-edge detector fusion
        # Sobel edge
        sobel_edges = kf.sobel(img_tensor.mean(dim=0, keepdim=True).unsqueeze(0))
        # Laplacian edge
        laplacian_edges = kf.laplacian(
            img_tensor.mean(dim=0, keepdim=True).unsqueeze(0), kernel_size=3
        )

        # Fusion multiple edges
        edges = 0.6 * sobel_edges + 0.4 * laplacian_edges

        edges = F.interpolate(
            edges,
            size=(cam_tensor.shape[0], cam_tensor.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        edges = edges.squeeze()  # [H, W]

        # Normalize edge intensity
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)

        # Adaptive edge threshold - based on edge distribution
        edge_threshold = (
            torch.quantile(edges.view(-1), 0.7) * 0.8
        )  # Take 70% quantile's 80% as threshold
        edge_weight = edges > edge_threshold  # Adaptive edge threshold

        # Edge-aware adjustment: sharpen CAM values at boundary
        cam_tensor_adjusted = cam_tensor.clone()

        # Enhance CAM contrast at edge - gradient enhancement
        edge_intensity = (edges - edge_threshold).clamp(0, 1) * 0.5  # Edge intensity
        cam_enhancement = 1.0 + edge_intensity  # Edge enhancement factor

        # Apply edge enhancement
        cam_tensor_adjusted = torch.where(
            edge_weight,
            torch.clamp(cam_tensor * cam_enhancement, 0, 1),  # Edge enhancement
            cam_tensor,  # Non-edge areas remain unchanged
        )

        # Update CAM
        ensemble_cam[i, 0] = cam_tensor_adjusted

        # Save result
        single_cam = ensemble_cam[i, 0].detach().cpu()  # (H,W)
        torch.save(single_cam, f"data/CAM/{image_ids[i]}.pt")

    # Create CRF model and move to GPU
    dense_crf = DenseCRF(iter_max=10).to(
        device
    )  # Reduce iterations but optimize other parameters

    # CRF Processing (per image in batch)
    for i in range(x.shape[0]):
        # Get image and CAM data, keep on GPU
        img_tensor = x[i].to(device)  # [3, H, W]
        cam_tensor = ensemble_cam[i, 0].to(device)  # [H, W]

        # Clean up GPU cache to reduce memory fragmentation
        torch.cuda.empty_cache()

        # Skip invalid images
        if torch.max(cam_tensor) < 0.1:
            continue

        # Image-specific adaptive adjustment of CRF parameters
        # Calculate image complexity - based on edge density
        img_gray = img_tensor.mean(dim=0)
        img_edges = kf.sobel(img_gray.unsqueeze(0).unsqueeze(0))
        edge_density = torch.mean((img_edges > 0.1).float())  # Convert boolean to float

        # Adapt parameters based on image complexity
        sxy_gaussian = (
            3 if edge_density > 0.1 else 5
        )  # Smaller Gaussian window for complex images
        compat_gaussian = (
            8 + 8 * edge_density
        )  # Increase compatibility parameter for complex images
        sxy_bilateral = 20 + 10 * (
            1 - edge_density
        )  # Larger bilateral window for simple images
        srgb_bilateral = (
            8 + 7 * edge_density
        )  # Increase color sensitivity for complex images
        compat_bilateral = (
            4 + 3 * edge_density
        )  # Increase compatibility for complex images

        # Prepare unary potentials (on GPU)
        unary = torch.stack([1 - cam_tensor, cam_tensor], dim=0)  # [2, H, W]

        # Execute CRF inference - use adaptive parameters
        Q = dense_crf(
            img_tensor,
            unary,
            sxy_gaussian=sxy_gaussian,
            compat_gaussian=compat_gaussian,
            sxy_bilateral=sxy_bilateral,
            srgb_bilateral=srgb_bilateral,
            compat_bilateral=compat_bilateral,
        )

        prob_foreground = Q[
            1
        ]  # Q is shape [2, H, W], take probability map of 1st class (foreground)

        torch.save(prob_foreground.cpu(), f"data/CAM/{image_ids[i]}.pt")


def get_cam(image_ids):
    cam = [
        torch.load(f"data/CAM/{image_id}.pt", map_location=device)
        .reshape(1, 1, 256, 256)
        .to(device)
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
