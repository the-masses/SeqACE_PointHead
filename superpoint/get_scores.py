import torch
import numpy as np
from pathlib import Path
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
from superpoint import SuperPoint

def load_image(image_path):
    transforms = Compose([Resize(480), ToTensor()])
    image = Image.open(image_path).convert('L')
    return transforms(image).unsqueeze(0)

def process_images(image_dir, model, keypoints_output_dir, scores_output_dir, dense_scores_output_dir, device='cuda'):
    image_paths = list(Path(image_dir).glob('*.[pj][np][g]'))
    image_paths.sort(key=lambda p: int(p.stem))
    Path(keypoints_output_dir).mkdir(parents=True, exist_ok=True)
    Path(scores_output_dir).mkdir(parents=True, exist_ok=True)
    Path(dense_scores_output_dir).mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        image = load_image(image_path).to(device)
        with torch.no_grad():
            output = model({'image': image})
            keypoints = torch.stack(output['keypoints'], dim=0).cpu().numpy()
            scores = torch.stack(output['scores'], dim=0).cpu().numpy()
            dense_scores = output['dense_scores'].astype(np.float16)
        file_name = image_path.stem
        np.save(Path(keypoints_output_dir, f'{file_name}.npy'), keypoints)
        np.save(Path(scores_output_dir, f'{file_name}.npy'), scores)
        np.save(Path(dense_scores_output_dir, f'{file_name}.npy'), dense_scores)

if __name__ == '__main__':
    config = {
        'descriptor_dim': 256,
        'nms_radius': 16,
        'keypoint_threshold': 0.015,
        'max_keypoints': 300,
        'remove_borders': 4,
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SuperPoint(config=config).to(device)
    model.eval()
    model.load_state_dict(torch.load('./weights/superpoint_v1.pth'))

    image_dir = '/home/jz/Documents/Re-localization/Code/dsacstar-pretrained/datasets/7scenes/stairs/test/rgb/'
    keypoints_output_dir = '../datasets/7scenes/stairs/test/keypoints/'
    scores_output_dir = '../datasets/7scenes/stairs/test/scores/'
    dense_scores_output_dir = '../datasets/7scenes/stairs/test/dense_scores/'

    process_images(image_dir, model, keypoints_output_dir, scores_output_dir, dense_scores_output_dir, device=device)

