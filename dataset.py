import os
import numpy as np
import math
import torch
import random

from PIL import Image
from skimage import io
from skimage import color
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.transform import warp
from scipy.spatial.transform import Rotation as R

from data_aug import add_fog, add_shade, motion_blur, additive_gaussian_noise, additive_speckle_noise, random_brightness, random_contrast, random_homography_transform


class RandomHomographyTransform3D(object):
    def __init__(self, focal_length, max_angle=40):
        self.focal_length = focal_length
        self.max_angle = max_angle
    
    def __call__(self, img, pose):
        image_np = np.array(img)
        height, width = image_np.shape[:2]

        K = np.array([
            [self.focal_length, 0, width / 2],
            [0, self.focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        angle_rad = np.deg2rad(np.random.uniform(-self.max_angle, self.max_angle))
        axis = np.random.rand(3) - 0.5
        rotation_3x3 = R.from_rotvec(angle_rad * axis).as_matrix()

        rotation_4x4 = np.eye(4, dtype=np.float32)
        rotation_4x4[:3, :3] = rotation_3x3

        H = K @ rotation_3x3 @ np.linalg.inv(K)
        H /= H[2, 2]

        warped_image = warp(image_np, H, output_shape=(height, width), preserve_range=True)
        warped_image = Image.fromarray(warped_image.astype(np.uint8))

        updated_pose = np.dot(pose, rotation_4x4)
        updated_pose = torch.from_numpy(updated_pose).float()

        return warped_image, updated_pose


class CamLocDataset(Dataset):
	"""Camera localization dataset.

	Access to image, calibration and ground truth data given a dataset directory.
	"""
	def __init__(self, root_dir, scene_idx, 
		augment=False,
		aug_scale_min=1/2,
		aug_scale_max=2,
		aug_contrast=0.4, 
		aug_brightness=0.4,
		aug_saturation=0.3,
    	aug_hue=0.3,
		image_height=480):
		'''Constructor.

		Parameters:
			root_dir: Folder of the data (training or test).
			aug_contrast: Max relative scale factor for image contrast sampling, e.g. 0.1 -> [0.9,1.1]
			aug_brightness: Max relative scale factor for image brightness sampling, e.g. 0.1 -> [0.9,1.1]
			image_height: RGB images are rescaled to this maximum height
		'''
		self.scene_idx = scene_idx
		self.image_height = image_height
		self.augment = augment
		self.aug_scale_min = aug_scale_min
		self.aug_scale_max = aug_scale_max
		self.aug_contrast = aug_contrast
		self.aug_brightness = aug_brightness
		self.aug_saturation = aug_saturation
		self.aug_hue = aug_hue


		rgb_dir = root_dir + '/rgb/'
		pose_dir =  root_dir + '/poses/'
		calibration_dir = root_dir + '/calibration/'
		dense_scores_dir = root_dir + '/dense_scores/'

		def sort_key(file):
			return int(os.path.splitext(os.path.basename(file))[0])
		
		# self.rgb_files = os.listdir(rgb_dir)
		# self.rgb_files = [rgb_dir + f for f in self.rgb_files]
		# self.rgb_files.sort()

		self.rgb_files = [os.path.join(rgb_dir, f) for f in sorted(os.listdir(rgb_dir), key=sort_key)]

		# self.pose_files = os.listdir(pose_dir)
		# self.pose_files = [pose_dir + f for f in self.pose_files]
		# self.pose_files.sort()
		self.pose_files = [os.path.join(pose_dir, f) for f in sorted(os.listdir(pose_dir), key=sort_key)]

		# self.calibration_files = os.listdir(calibration_dir)
		# self.calibration_files = [calibration_dir + f for f in self.calibration_files]
		# self.calibration_files.sort()		
		self.calibration_files = [os.path.join(calibration_dir, f) for f in sorted(os.listdir(calibration_dir), key=sort_key)]

		self.dense_scores_files = [os.path.join(dense_scores_dir, f) for f in sorted(os.listdir(dense_scores_dir), key=sort_key)]

		if len(self.rgb_files) != len(self.pose_files):
			raise Exception('RGB file count does not match pose file count!')
		
		if len(self.rgb_files) != len(self.calibration_files):
			raise RuntimeError('RGB file count does not match calibration file count!')
		
		if len(self.rgb_files) != len(self.dense_scores_files):
			raise RuntimeError('RGB file count does not match dense scores file count!')

	def __len__(self):
		return len(self.rgb_files)

	def __getitem__(self, idx):

		image = io.imread(self.rgb_files[idx])

		if len(image.shape) < 3:
			image = color.gray2rgb(image)

		focal_length = float(np.loadtxt(self.calibration_files[idx]))

		# image will be normalized to standard height, adjust focal length as well
		f_scale_factor = self.image_height / image.shape[0]
		# print(f'f_scale_factor: {f_scale_factor}')
		# print(f'image_height:{self.image_height}')
		# print(f'image.shape[0]: {image.shape[0]}')
		focal_length *= f_scale_factor
		# print(f'focal_length: {focal_length}')

		pose = np.loadtxt(self.pose_files[idx])
		pose = torch.from_numpy(pose).float()

		# keypoints = np.load(self.keypoints_files[idx]).squeeze(0)
		# scores = np.load(self.scores_files[idx]).squeeze(0)

		# heatmap = np.zeros((480, 640))
		# for i in range(keypoints.shape[0]):
		# 	x, y = keypoints[i]  
		# 	score = scores[i]
		# 	if 0 <= x < 640 and 0 <= y < 480:
		# 		heatmap[int(y), int(x)] = score

		# heatmap = torch.from_numpy(heatmap)
		
		# Supervised by Superpoint
		scores_label = np.load(self.dense_scores_files[idx])		# Dense_scores_files are float16 documents.
		scores_label = scores_label.astype(np.float32)				# Need to change to float32.

		if self.augment:

			# scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)
			augmentation_methods = [
				add_shade,
				add_fog,
				motion_blur,
				additive_gaussian_noise,
				additive_speckle_noise,
				random_brightness,
				random_contrast
			]
			augmentation_method = random.choice(augmentation_methods)
			image = augmentation_method(image)

			# augment input image
			self.image_transform = transforms.Compose([
				transforms.ToPILImage(),
				transforms.Resize(int(self.image_height)),
				transforms.ColorJitter(
					brightness=self.aug_brightness,
    				contrast=self.aug_contrast,
    				saturation=self.aug_saturation,
    				hue=self.aug_hue
				),
				transforms.Grayscale(),
				transforms.ToTensor(),
				transforms.Normalize(
					mean=[0.4],
					std=[0.25]
					)
			])
			image = self.image_transform(image)

			# self.image_transform_aug = transforms.Compose([
			# 	transforms.ToPILImage(),
			# 	transforms.Resize(int(self.image_height)),
			# 	transforms.ColorJitter(
			# 		brightness=self.aug_brightness,
			# 		contrast=self.aug_contrast,
			# 		saturation=self.aug_saturation,
			# 		hue=self.aug_hue
			# 	),
			# 	transforms.Grayscale(),
			# 	transforms.ToTensor(),
			# 	transforms.Normalize(
			# 		mean=[0.4],
			# 		std=[0.25]
			# 	)
			# ])
			# image = self.image_transform_aug(image)
			# homography_transform = RandomHomographyTransform3D(focal_length=focal_length)
			# image, pose = homography_transform(image, pose)

			# Save augmented images for check
			# save_path = os.path.join(save_dir, f'augmented_image_{idx}.png')
			# image_pil = transforms.ToPILImage()(image.squeeze(0))  # 为保存去掉通道维度
			# image_pil.save(save_path)

		else:
			self.image_transform = transforms.Compose([
				transforms.ToPILImage(),
				transforms.Resize(self.image_height),
				transforms.Grayscale(),
				transforms.ToTensor(),
				transforms.Normalize(
				mean=[0.4], # statistics calculated over 7scenes training set, should generalize fairly well
				std=[0.25]
				)
			])
			image = self.image_transform(image)	
			# print(f'focal_length: {focal_length}')
			# print(f'pose: {pose}')

		return image, pose, focal_length, self.rgb_files[idx], scores_label
	
class MultiCamLocDataset(Dataset):
	def __init__(self, dataset_configs):
		self.datasets = [CamLocDataset(root_dir, scene_idx, 
										augment=False,
										aug_scale_min=1/2,
										aug_scale_max=2,
										aug_contrast=0.4,
										aug_brightness=0.4,
										aug_saturation=0.3,
    									aug_hue=0.3,
										image_height=480)for root_dir, scene_idx in dataset_configs]

	def __len__(self):
		return max(len(d) for d in self.datasets)
	
	def __getitem__(self, idx):
			results = []
			for dataset in self.datasets:
				idx_mod = idx % len(dataset)
				result = dataset[idx_mod]
				results.append(result)

			images, poses, focal_lengths, file_names, scores_label = zip(*results)
			scene_indices = [dataset.scene_idx for dataset in self.datasets]
			# for focal_lengths, file_name, scene_idx in zip(focal_lengths, file_names, scene_indices):
			# 	print(f"Focal_lengths: {focal_lengths}, File: {file_name}, Scene Index: {scene_idx}")

			return images, poses, focal_lengths, file_names, scene_indices, scores_label

		