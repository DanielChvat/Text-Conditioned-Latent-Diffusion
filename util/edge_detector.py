import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F


def create_gaussain_kernel(n=5, sigma=1.5):
    ax = torch.arange(n) - 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / (kernel.sum())
    return kernel


class EdgeDetector(nn.Module):
    def __init__(self, n=5, sigma=1.5):
        super().__init__()

        g = create_gaussain_kernel(n, sigma)

        self.blur_horizontal = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=n, padding=2, bias=False
        )
        self.blur_vertical = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=n, padding=2, bias=False
        )

        self.blur_horizontal.weight.data.copy_(g)
        self.blur_horizontal.weight.requires_grad_(False)
        self.blur_vertical.weight.data.copy_(g.T)
        self.blur_vertical.weight.requires_grad_(False)

        sobel_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_x.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_y.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_x.weight.requires_grad_(False)
        self.sobel_y.weight.requires_grad_(False)

        filter_0 = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
        filter_45 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
        filter_90 = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
        filter_135 = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]])
        filter_180 = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
        filter_225 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
        filter_270 = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
        filter_315 = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])

        directional_filters = np.stack(
            [
                filter_0,
                filter_45,
                filter_90,
                filter_135,
                filter_180,
                filter_225,
                filter_270,
                filter_315,
            ]
        )
        self.directional_filter = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False
        )
        self.directional_filter.weight.data.copy_(
            torch.from_numpy(directional_filters[:, None, ...])
        )
        self.directional_filter.weight.requires_grad_(False)

        self.local_mean = nn.AvgPool2d(15, stride=1, padding=7)
        self.local_var = nn.AvgPool2d(15, stride=1, padding=7)

    def forward(self, img):
        img_r = img[:, 0:1]
        img_g = img[:, 1:2]
        img_b = img[:, 2:3]

        blurred_r = self.blur_vertical(self.blur_horizontal(img_r))
        blurred_g = self.blur_vertical(self.blur_horizontal(img_g))
        blurred_b = self.blur_vertical(self.blur_horizontal(img_b))

        pre_gx = self.sobel_x(blurred_r)
        pre_gy = self.sobel_y(blurred_r)
        pre_grad = torch.hypot(pre_gx, pre_gy)

        mu_r = self.local_mean(blurred_r)
        mu_g = self.local_mean(blurred_g)
        mu_b = self.local_mean(blurred_b)

        detail_r = blurred_r - mu_r
        detail_g = blurred_g - mu_g
        detail_b = blurred_b - mu_b

        w = torch.clamp(pre_grad, 0, 1)

        blurred_r = blurred_r + w * detail_r
        blurred_g = blurred_g + w * detail_g
        blurred_b = blurred_b + w * detail_b

        grad_x_r = self.sobel_x(blurred_r)
        grad_y_r = self.sobel_y(blurred_r)

        grad_x_g = self.sobel_x(blurred_g)
        grad_y_g = self.sobel_y(blurred_g)

        grad_x_b = self.sobel_x(blurred_b)
        grad_y_b = self.sobel_y(blurred_b)

        gx = torch.sqrt(grad_x_r**2 + grad_x_g**2 + grad_x_b**2)
        gy = torch.sqrt(grad_y_r**2 + grad_y_g**2 + grad_y_b**2)

        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-8) ** 0.5

        # flat = grad_mag.view(grad_mag.shape[0], -1)
        # p99 = flat.kthvalue(int(0.99 * flat.shape[1]), dim=1)[0].view(grad_mag.shape[0], 1, 1, 1)
        # p99 = torch.clamp(p99, min=1e-6)
        # edge_map = torch.clamp(grad_mag / (p99 + 1e-8), 0, 1)

        edge_map = torch.log1p(grad_mag)

        blurred_img = torch.cat([blurred_r, blurred_g, blurred_b], dim=1)

        return blurred_img, edge_map


if __name__ == "__main__":
    img = Image.open("./ImageNet100/train/n01484850/n01484850_10016.JPEG").convert(
        "RGB"
    )
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    img_tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(
        img_tensor
    )

    net = EdgeDetector(n=5, sigma=1.5)
    net.eval()

    with torch.no_grad():
        blurred_img, grad_mag = net(img_tensor)
        print("kernel sum:", net.blur_horizontal.weight.data.sum())
        print("grad_mag max:", grad_mag.max().item())
        print("grad_mag mean:", grad_mag.mean().item())

    grad_mag_np = grad_mag.squeeze().cpu().numpy()

    plt.figure(figsize=(8, 2))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Edges Magnitude")
    plt.imshow(grad_mag_np, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
