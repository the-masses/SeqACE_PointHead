import cv2
import os
import numpy as np
import glob
from pathlib import Path

# My network output datasets
keypoints_dir = './results_keypoint/keypoints/'
image_dir = './datasets/ScanNet/scene0000_00/rgb/'
output_dir = './results_keypoint/keypoint_images_0/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Function to get file paths for images considering both jpg and png formats
def get_image_files(directory):
    return sorted(glob.glob(os.path.join(directory, '*.jpg')) + glob.glob(os.path.join(directory, '*.png')),
                  key=lambda x: int(Path(x).stem))

keypoints_files = sorted(glob.glob(os.path.join(keypoints_dir, '*.npy')), key=lambda x: int(Path(x).stem))
images_files = get_image_files(image_dir)

for keypoints_path, images_path in zip(keypoints_files, images_files):
    keypoints = np.load(keypoints_path).squeeze()
    print(keypoints.shape)
    images = cv2.imread(images_path)
    resized_images = cv2.resize(images, (640, 480))
    for c in keypoints:
        x, y = c.astype(int)
        cv2.circle(resized_images, (x, y), radius=3, color=(255, 255, 0), thickness=-1)
    output_path = os.path.join(output_dir, f'{Path(images_path).stem}.jpg')
    cv2.imwrite(output_path, resized_images)

print("\n===================================================")
print("\nComplete.")