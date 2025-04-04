import os
import json
import numpy as np
import imageio.v2 as imageio
import torch
from torch.utils.data import Dataset
from typing import Tuple


def load_dataset(dataset_path: str,
                 mode: str = 'train',
                 single_image: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load images and camera-to-world transformation matrices from the dataset.

    This function reads a JSON file containing camera parameters and frame information,
    loads each corresponding image, and composites images with an alpha channel over a 
    white background. It also computes the camera's focal length based on the horizontal
    field of view.

    Args:
        dataset_path (str): Base directory of the dataset.
        mode (str): Dataset split mode ('train', 'val', or 'test').

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - images: Array of shape (N, H, W, 3) with normalized RGB images.
            - c2w_matrices: Array of shape (N, 4, 4) with camera-to-world transformation matrices.
            - focal_length: Focal length computed from the camera's field of view.
    """
    transforms_path = os.path.join(dataset_path, f"transforms_{mode}.json")
    with open(transforms_path, 'r') as f:
        meta = json.load(f)

    camera_angle_x = meta["camera_angle_x"]
    frames = meta["frames"]

    images = []
    c2w_matrices = []
    for frame in frames:
        rel_path = frame["file_path"].lstrip("./")
        img_path = os.path.join(dataset_path, rel_path + ".png")
        img = imageio.imread(img_path).astype(np.float32) / 255.0
        
        # Composite image with alpha channel over white background
        if img.shape[-1] == 4:
            alpha = img[..., 3:4]
            img = img[..., :3] * alpha + (1.0 - alpha)
        
        images.append(img)
        c2w_matrices.append(np.array(frame["transform_matrix"], dtype=np.float32))
        
        if single_image:
            break
    
    images = np.stack(images, axis=0)
    c2w_matrices = np.stack(c2w_matrices, axis=0)
    _, _, W, _ = images.shape

    focal_length = 0.5 * W / np.tan(0.5 * camera_angle_x)
    
    return images, c2w_matrices, focal_length


def compute_rays(images: np.ndarray,
                 c2w_matrices: np.ndarray,
                 focal_length: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute camera ray origins and directions, and extract target pixel colors.

    This function generates a meshgrid of pixel coordinates, converts these coordinates 
    into camera space, and applies the camera-to-world rotation and translation to compute 
    ray origins and normalized ray directions for each pixel. It also flattens the target 
    pixel colors for subsequent processing.

    Args:
        images (np.ndarray): Array of shape (N, H, W, 3) with RGB images.
        c2w_matrices (np.ndarray): Array of shape (N, 4, 4) with camera-to-world matrices.
        focal_length (float): Focal length derived from the camera intrinsics.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - rays_o: Ray origins with shape (N, H*W, 3).
            - rays_d: Normalized ray directions with shape (N, H*W, 3).
            - target_pixels: Flattened RGB pixel colors with shape (N, H*W, 3).
    """
    N, H, W, _ = images.shape

    target_pixels = images.reshape(N, -1, 3)

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v, indexing='xy')

    # Convert pixel coordinates to camera space
    x_cam = u_grid - 0.5 * W
    y_cam = -(v_grid - 0.5 * H)
    z_cam = -np.full_like(x_cam, focal_length)
    directions_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

    R = c2w_matrices[:, :3, :3]
    t = c2w_matrices[:, :3, 3]

    # Rotate ray directions to world space
    rays_d = np.einsum('nij,hwj->nhwi', R, directions_cam)
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)

    # Replicate camera origin for each ray
    rays_o = np.tile(t[:, None, None, :], (1, H, W, 1))

    rays_o = rays_o.reshape(N, -1, 3)
    rays_d = rays_d.reshape(N, -1, 3)

    return rays_o, rays_d, target_pixels


class RayDataset(Dataset):
    """
    A PyTorch dataset for storing and accessing ray data.

    This dataset class converts ray origins, ray directions, and target pixel colors from 
    NumPy arrays to torch tensors. It flattens the input arrays to produce a dataset 
    where each sample corresponds to a single ray along with its associated color.

    Args:
        rays_o (np.ndarray): Array of ray origins with shape (N, H*W, 3).
        rays_d (np.ndarray): Array of ray directions with shape (N, H*W, 3).
        target_pixels (np.ndarray): Array of target RGB pixel colors with shape (N, H*W, 3).
    """
    def __init__(self, rays_o, rays_d, target_pixels):
        self.rays_o = torch.from_numpy(rays_o.reshape(-1, 3)).float()
        self.rays_d = torch.from_numpy(rays_d.reshape(-1, 3)).float()
        self.target_pixels = torch.from_numpy(target_pixels.reshape(-1, 3)).float()

    def __len__(self):
        return self.rays_o.shape[0]

    def __getitem__(self, idx):
        return self.rays_o[idx], self.rays_d[idx], self.target_pixels[idx]
