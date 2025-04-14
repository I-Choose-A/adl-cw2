import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import kornia.filters as kf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        h, w = img.shape[1], img.shape[2]
        n_labels = unary.shape[0]

        # use smaller image size to avoid memory issues
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

            # process the smaller version directly
            Q_small = self._dense_crf(
                img_small,
                unary_small,
                sxy_gaussian,
                compat_gaussian,
                sxy_bilateral,
                srgb_bilateral,
                compat_bilateral,
            )

            # upsample back to original size
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

    # actual CRF computation for smaller images
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
        h, w = img.shape[1], img.shape[2]
        n_labels = unary.shape[0]
        n_pixels = h * w

        # initialize Q with unary potentials
        Q = unary.clone()

        # use gaussian filter approach to avoid full pairwise matrices
        for _ in range(self.iter_max):
            # apply softmax to Q
            Q = F.softmax(Q, dim=0)

            # prepare messages
            message = torch.zeros_like(Q)

            # spatial gaussian message passing - using separable convolution approximation
            for label in range(n_labels):
                # use gaussian blur to approximate message passing
                feat = Q[label: label + 1]  # [1, H, W]
                # filter size proportional to sigma
                ksize = int(2 * sxy_gaussian) * 2 + 1
                # apply separable gaussian filtering
                blurred = self._gaussian_blur(feat, ksize, sxy_gaussian)
                # compute message
                message[label: label + 1] += compat_gaussian * (blurred - feat)

            # bilateral message passing - using guided filter approximation
            for label in range(n_labels):
                # use guided filter to approximate bilateral filter
                feat = Q[label: label + 1]  # [1, H, W]
                guided = self._guided_filter(img, feat, sxy_bilateral, srgb_bilateral)
                message[label: label + 1] += compat_bilateral * (guided - feat)

            # update Q
            Q = unary - message

        # final Q
        Q = F.softmax(Q, dim=0)
        return Q

    # implement gaussian blur using separable convolution
    def _gaussian_blur(self, x, kernel_size, sigma):
        # create 1D gaussian kernel
        grid = (
                torch.arange(kernel_size, device=x.device).float() - (kernel_size - 1) / 2
        )
        gaussian = torch.exp(-(grid ** 2) / (2 * sigma ** 2))
        kernel = gaussian / gaussian.sum()

        # apply separable convolution
        padding = (kernel_size - 1) // 2

        # create convolution weights
        weight_h = kernel.view(1, 1, kernel_size, 1).repeat(1, 1, 1, 1)
        weight_w = kernel.view(1, 1, 1, kernel_size).repeat(1, 1, 1, 1)

        # horizontal convolution
        x = F.conv2d(x, weight_h, padding=(padding, 0), groups=1)
        # vertical convolution
        x = F.conv2d(x, weight_w, padding=(0, padding), groups=1)
        return x

    def _guided_filter(self, guide, src, radius, eps):
        # record original dimensions for later processing
        guide_dim = guide.dim()
        src_dim = src.dim()

        # ensure guide and src are 4D tensors [N,C,H,W]
        if guide_dim == 3:  # [C,H,W]
            guide = guide.unsqueeze(0)
        elif guide_dim == 2:  # [H,W]
            guide = guide.unsqueeze(0).unsqueeze(0)

        if src_dim == 3:  # [C,H,W]
            src = src.unsqueeze(0)
        elif src_dim == 2:  # [H,W]
            src = src.unsqueeze(0).unsqueeze(0)

        # ensure guide and src have same spatial dimensions
        if guide.shape[2:] != src.shape[2:]:
            src = F.interpolate(
                src,
                size=(guide.shape[2], guide.shape[3]),
                mode="bilinear",
                align_corners=False,
            )

        # convert multi-channel guide to single channel
        if guide.shape[1] > 1:
            guide = guide.mean(dim=1, keepdim=True)  # [N, 1, H, W]

        # mean filter
        kernel_size = int(2 * radius) + 1
        padding = (kernel_size - 1) // 2

        # create mean convolution kernel
        mean_kernel = torch.ones(
            1, 1, kernel_size, kernel_size, device=guide.device
        ) / (kernel_size ** 2)

        # compute means
        mean_guide = F.conv2d(guide, mean_kernel, padding=padding)
        mean_src = F.conv2d(src, mean_kernel, padding=padding)
        mean_guide_src = F.conv2d(guide * src, mean_kernel, padding=padding)

        # compute covariance
        cov_guide_src = mean_guide_src - mean_guide * mean_src

        # compute variance
        var_guide = (
                F.conv2d(guide * guide, mean_kernel, padding=padding) - mean_guide ** 2
        )

        # compute a and b
        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide

        # compute mean a and mean b
        mean_a = F.conv2d(a, mean_kernel, padding=padding)
        mean_b = F.conv2d(b, mean_kernel, padding=padding)

        # explicitly check and handle dimensions
        target_size = (guide.shape[2], guide.shape[3])  # explicitly specified as (H,W) tuple

        # ensure dimensions match
        if mean_a.shape[2:] != guide.shape[2:]:
            mean_a = F.interpolate(
                mean_a, size=target_size, mode="bilinear", align_corners=False
            )

        if mean_b.shape[2:] != guide.shape[2:]:
            mean_b = F.interpolate(
                mean_b, size=target_size, mode="bilinear", align_corners=False
            )

        # final output
        output = mean_a * guide + mean_b

        # restore original dimensions
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

    # data augmentation to generate multiple views
    transforms_list = [
        # original image
        lambda img: img,
        # horizontal flip
        lambda img: torch.flip(img, dims=[-1]),
        # rotate 10 degrees
        lambda img: torch.nn.functional.affine_grid(
            torch.tensor([[[1, 0, 0], [0, 1, 0]]], device=device)
            * torch.cos(torch.tensor(10 * np.pi / 180)),
            size=torch.Size((1, img.shape[0], img.shape[1], img.shape[2])),
            align_corners=False,
        ).to(device),
        # scale by 0.9
        lambda img: F.interpolate(
            img.unsqueeze(0), scale_factor=0.9, mode="bilinear", align_corners=False
        )[0],
    ]

    # store CAMs for all augmented versions
    all_cams = []

    # generate CAM for each augmented version
    for transform_fn in transforms_list:
        # skip complex transforms (only keep first two simple transforms for demo)
        if transform_fn == transforms_list[2] or transform_fn == transforms_list[3]:
            continue

        # apply transform
        transformed_x = torch.stack([transform_fn(img) for img in x])

        # collect features from different layers
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

        # register forward and backward hooks
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

        # forward pass
        logits = model(transformed_x)

        model.zero_grad()  # clear gradients

        # only compute gradients for target class
        one_hot_y = F.one_hot(y, num_classes=37).to(device)
        logits.backward(gradient=one_hot_y, retain_graph=True)

        # remove hooks
        handle_layer1.remove()
        handle_layer2.remove()
        handle_layer3.remove()
        handle_layer4.remove()
        handle_grad_layer1.remove()
        handle_grad_layer2.remove()
        handle_grad_layer3.remove()
        handle_grad_layer4.remove()

        # only use data from first call
        features_layer1 = features_layer1[0]  # shape = (B,C,H,W)
        features_layer2 = features_layer2[0]  # shape = (B,C,H,W)
        features_layer3 = features_layer3[0]  # shape = (B,C,H,W)
        features_layer4 = features_layer4[0]  # shape = (B,C,H,W)
        gradients_layer1 = gradients_layer1[0]  # shape = (B,C,H,W)
        gradients_layer2 = gradients_layer2[0]  # shape = (B,C,H,W)
        gradients_layer3 = gradients_layer3[0]  # shape = (B,C,H,W)
        gradients_layer4 = gradients_layer4[0]  # shape = (B,C,H,W)

        # add channel attention mechanism
        def channel_attention(features, gradients):
            # compute channel importance
            channel_weights = torch.sum(torch.abs(gradients), dim=[2, 3], keepdim=True)
            channel_weights = F.softmax(channel_weights, dim=1)
            # apply channel attention
            weighted_features = features * channel_weights
            return weighted_features

        # apply channel attention to all layers
        features_layer1 = channel_attention(features_layer1, gradients_layer1)
        features_layer2 = channel_attention(features_layer2, gradients_layer2)
        features_layer3 = channel_attention(features_layer3, gradients_layer3)
        features_layer4 = channel_attention(features_layer4, gradients_layer4)

        # compute GradCAM++ for all layers separately
        # define a function to calculate GradCAM++
        def calculate_gradcam_pp(features, gradients):
            grad_2 = gradients ** 2
            grad_3 = grad_2 * gradients

            alpha_num = grad_2
            alpha_denom = 2 * grad_2 + features * grad_3 + 1e-8

            alpha = alpha_num / alpha_denom
            alpha_norm = alpha / (torch.sum(alpha, dim=[1, 2, 3], keepdim=True) + 1e-8)

            cam = torch.sum(alpha_norm * F.relu(features), dim=1)
            cam = F.relu(cam)
            return cam

        # compute CAM for each layer
        cam_layer1 = calculate_gradcam_pp(features_layer1, gradients_layer1)
        cam_layer2 = calculate_gradcam_pp(features_layer2, gradients_layer2)
        cam_layer3 = calculate_gradcam_pp(features_layer3, gradients_layer3)
        cam_layer4 = calculate_gradcam_pp(features_layer4, gradients_layer4)

        # resize all CAMs to same size (using layer4 size as reference)
        target_size = cam_layer4.shape[1:]

        # upsample other layers' CAMs to target_size
        def upsample_cam(cam, target_size):
            return F.interpolate(
                cam.unsqueeze(1), size=target_size, mode="bilinear", align_corners=False
            ).squeeze(1)

        cam_layer1 = upsample_cam(cam_layer1, target_size)
        cam_layer2 = upsample_cam(cam_layer2, target_size)
        cam_layer3 = upsample_cam(cam_layer3, target_size)

        # normalize all layers' CAMs
        def normalize_cam(cam):
            return (cam - cam.amin(dim=(1, 2), keepdim=True)[0]) / (
                    cam.amax(dim=(1, 2), keepdim=True)[0] + 1e-8
            )

        cam_layer1 = normalize_cam(cam_layer1)
        cam_layer2 = normalize_cam(cam_layer2)
        cam_layer3 = normalize_cam(cam_layer3)
        cam_layer4 = normalize_cam(cam_layer4)

        # compute layer clarity metric
        def calculate_clarity(cam):
            # use gradient magnitude as clarity metric
            grad_x = torch.abs(cam[:, :, 1:] - cam[:, :, :-1])
            grad_y = torch.abs(cam[:, 1:, :] - cam[:, :-1, :])

            # average gradient magnitude
            clarity = (torch.mean(grad_x) + torch.mean(grad_y)) / 2
            return clarity

        clarity_layer1 = calculate_clarity(cam_layer1)
        clarity_layer2 = calculate_clarity(cam_layer2)
        clarity_layer3 = calculate_clarity(cam_layer3)
        clarity_layer4 = calculate_clarity(cam_layer4)

        # compute adaptive weights
        total_clarity = (
                clarity_layer1 + clarity_layer2 + clarity_layer3 + clarity_layer4 + 1e-8
        )

        # base weights - biased toward higher-level features
        base_weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device)

        # clarity weights
        clarity_weights = (
                torch.tensor(
                    [clarity_layer1, clarity_layer2, clarity_layer3, clarity_layer4],
                    device=device,
                )
                / total_clarity
        )

        # fuse base and clarity weights
        alpha = 0.7  # controls influence of base weights
        fusion_weights = alpha * base_weights + (1 - alpha) * clarity_weights

        # normalize weight sum
        fusion_weights = fusion_weights / fusion_weights.sum()

        # apply multi-layer fusion
        cam = (
                fusion_weights[0] * cam_layer1
                + fusion_weights[1] * cam_layer2
                + fusion_weights[2] * cam_layer3
                + fusion_weights[3] * cam_layer4
        )

        # normalize again and continue previous processing
        cam = normalize_cam(cam)

        # upsample to input size
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # if it's horizontally flipped data, flip it back
        if transform_fn == transforms_list[1]:
            cam = torch.flip(cam, dims=[-1])

        # add to list
        all_cams.append(cam)

    # ensemble CAMs from multiple views (take average)
    ensemble_cam = torch.mean(torch.cat(all_cams, dim=1), dim=1, keepdim=True)

    # improved edge-aware enhancement
    for i in range(x.shape[0]):
        # get original CAM and image
        img_tensor = x[i]  # [3, H, W]
        cam_tensor = ensemble_cam[i, 0]  # [H, W]

        # multi-edge detector fusion
        # sobel edges
        sobel_edges = kf.sobel(img_tensor.mean(dim=0, keepdim=True).unsqueeze(0))
        # laplacian edges
        laplacian_edges = kf.laplacian(
            img_tensor.mean(dim=0, keepdim=True).unsqueeze(0), kernel_size=3
        )

        # fuse multiple edges
        edges = 0.6 * sobel_edges + 0.4 * laplacian_edges

        edges = F.interpolate(
            edges,
            size=(cam_tensor.shape[0], cam_tensor.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        edges = edges.squeeze()  # [H, W]

        # normalize edge strength
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)

        # adaptive edge threshold - based on edge distribution
        edge_threshold = (
                torch.quantile(edges.view(-1), 0.7) * 0.8
        )  # take 80% of 70th percentile as threshold
        edge_weight = edges > edge_threshold  # adaptive edge threshold

        # edge-aware adjustment: sharpen CAM values at boundaries
        cam_tensor_adjusted = cam_tensor.clone()

        # increase CAM contrast at edges - gradient enhancement
        edge_intensity = (edges - edge_threshold).clamp(0, 1) * 0.5  # edge strength
        cam_enhancement = 1.0 + edge_intensity  # enhancement factor at edges

        # apply edge enhancement
        cam_tensor_adjusted = torch.where(
            edge_weight,
            torch.clamp(cam_tensor * cam_enhancement, 0, 1),  # enhance at edges
            cam_tensor,  # keep unchanged elsewhere
        )

        # update CAM
        ensemble_cam[i, 0] = cam_tensor_adjusted

        # save results
        single_cam = ensemble_cam[i, 0].detach().cpu()  # (H,W)
        torch.save(single_cam, f"data/CAM/{image_ids[i]}.pt")

    # create CRF model and move to GPU
    dense_crf = DenseCRF(iter_max=10).to(device)  # reduce iterations but optimize other params

    # CRF processing (per image in batch)
    for i in range(x.shape[0]):
        # get image and CAM data, keep on GPU
        img_tensor = x[i].to(device)  # [3, H, W]
        cam_tensor = ensemble_cam[i, 0].to(device)  # [H, W]

        # clear GPU cache to reduce memory fragmentation
        torch.cuda.empty_cache()

        # skip invalid images
        if torch.max(cam_tensor) < 0.1:
            continue

        # adapt CRF params based on image characteristics
        # compute image complexity - based on edge density
        img_gray = img_tensor.mean(dim=0)
        img_edges = kf.sobel(img_gray.unsqueeze(0).unsqueeze(0))
        edge_density = torch.mean((img_edges > 0.1).float())  # convert bool to float

        # adapt params based on image complexity
        sxy_gaussian = 3 if edge_density > 0.1 else 5  # complex images use smaller gaussian window
        compat_gaussian = 8 + 8 * edge_density  # complex images increase compatibility param
        sxy_bilateral = 20 + 10 * (1 - edge_density)  # simple images use larger bilateral window
        srgb_bilateral = 8 + 7 * edge_density  # complex images increase color sensitivity
        compat_bilateral = 4 + 3 * edge_density  # complex images increase compatibility

        # prepare unary potentials (on GPU)
        unary = torch.stack([1 - cam_tensor, cam_tensor], dim=0)  # [2, H, W]

        # perform CRF inference - with adaptive params
        Q = dense_crf(
            img_tensor,
            unary,
            sxy_gaussian=sxy_gaussian,
            compat_gaussian=compat_gaussian,
            sxy_bilateral=sxy_bilateral,
            srgb_bilateral=srgb_bilateral,
            compat_bilateral=compat_bilateral,
        )

        # get final result
        refined = torch.argmax(Q, dim=0)  # [H, W]

        # save result
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
        trimap_path = f"../annotations/trimaps/{image_id}.png"
        trimap = Image.open(trimap_path)
        trimap = transforms.Resize((256, 256))(trimap)

        trimap = np.array(trimap)
        trimap[trimap == 2] = 0
        trimap[trimap == 3] = 1
        trimap = torch.from_numpy(trimap).float().reshape(1, 1, 256, 256)

        trimaps.append(trimap)
    return torch.cat(trimaps, dim=0)
