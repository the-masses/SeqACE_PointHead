import os
import time
import torch
import numpy as np
import cv2
from pathlib import Path
from skimage import io, color
from torchvision.transforms import transforms
from PIL import Image
from superpoint import SuperPoint


def load_image(image_path):
    """Load and preprocess image."""
    image = io.imread(image_path)
    if len(image.shape) < 3:
        image = color.gray2rgb(image)
    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((480, 640)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    image = image_transform(image)
    return image.unsqueeze(0)


def draw_keypoints(image_path, keypoints, output_path):
    """Draw keypoints and save image."""
    img = cv2.imread(str(image_path))
    img_resized = cv2.resize(img, (640, 480))
    for pt in keypoints:
        x, y = map(int, pt)
        cv2.circle(img_resized, (x, y), 3, (0, 255, 0), -1)
    cv2.imwrite(str(output_path), img_resized)


def process_images(image_dir, model, kp_out_dir, sc_out_dir, dense_out_dir, vis_out_dir, device='cuda'):
    image_paths = sorted(
        [p for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.ppm') for p in Path(image_dir).glob(ext)],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
    )

    for d in [kp_out_dir, sc_out_dir, dense_out_dir, vis_out_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    frame_times = []
    for image_path in image_paths:
        start = time.perf_counter()
        image = load_image(image_path).to(device)
        with torch.no_grad():
            output = model({'image': image})
            keypoints = torch.stack(output['keypoints'], dim=0).cpu().numpy().squeeze()
            scores = torch.stack(output['scores'], dim=0).cpu().numpy().squeeze()
            dense_scores = output['dense_scores']
            dense_scores = dense_scores.cpu().numpy().astype(np.float16) if isinstance(dense_scores, torch.Tensor) else dense_scores.astype(np.float16)

        stem = image_path.stem
        np.save(Path(kp_out_dir, f"{stem}.npy"), keypoints)
        np.save(Path(sc_out_dir, f"{stem}.npy"), scores)
        np.save(Path(dense_out_dir, f"{stem}.npy"), dense_scores)

        draw_keypoints(image_path, keypoints, Path(vis_out_dir, f"{stem}.jpg"))

        frame_time = time.perf_counter() - start
        frame_times.append(frame_time)
        print(f"[{stem}] Processed in {frame_time:.2f}s")

    if frame_times:
        print(f"Avg time/frame: {np.mean(frame_times) * 1000:.2f} ms")


if __name__ == '__main__':
    config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.0005,
        'max_keypoints': 300,
        'remove_borders': 4,
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SuperPoint(config=config).to(device).eval()
    print("Loading Weights...")
    # model.load_state_dict(torch.load('./weights/superpoint_v1.pth'))
    ckpt = torch.load('./weights/superpoint_v1.pth',
                  map_location='cpu')
    state_dict = ckpt.get('state_dict',
                      ckpt.get('model_state_dict'))
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print("Finished loading weights.")

    image_root = input("Input image root: ").strip()
    output_root = input("Input save path ").strip()

    kp_out_dir = Path(output_root) / 'keypoints'
    sc_out_dir = Path(output_root) / 'scores'
    dense_out_dir = Path(output_root) / 'dense_scores'
    vis_out_dir = Path(output_root) / 'vis'

    print(f"Begin processing images: {image_root}")
    process_images(
        image_dir=image_root,
        model=model,
        kp_out_dir=kp_out_dir,
        sc_out_dir=sc_out_dir,
        dense_out_dir=dense_out_dir,
        vis_out_dir=vis_out_dir,
        device=device
    )

