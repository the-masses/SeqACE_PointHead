import torch
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import argparse
import math

from dataset import MultiCamLocDataset
from network import Network
from superpoint.superpoint import simple_nms, remove_borders, top_k_keypoints

parser = argparse.ArgumentParser(
	description='Test a trained network on a specific scene.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('network', help='file name of a network trained for the scene')

parser.add_argument('--hypotheses', '-hyps', type=int, default=64, 
	help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--threshold', '-t', type=float, default=10, 
	help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

parser.add_argument('--inlieralpha', '-ia', type=float, default=100, 
	help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')

parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100, 
	help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')

parser.add_argument('--session', '-sid', default='',
	help='custom session name appended to output files, useful to separate different runs of a script')

opt = parser.parse_args()


base_dir = './datasets/ScanNet/'
scene_idx = 1
dataset_configs = [
	(f'{base_dir}scene{str(i).zfill(4)}_00', i) for i in range(scene_idx)
]

testset = MultiCamLocDataset(dataset_configs)
testset_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=6) # batch_size must be 1

# load network
network = Network(num_datasets=len(dataset_configs))
network.load_state_dict(torch.load(opt.network))
network = network.cuda()
network.eval()


print('Test images found: ', len(testset))
output_folder_keypoints = './results_keypoint'
os.makedirs(output_folder_keypoints, exist_ok=True)
counter = 0
default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1000,
        'remove_borders': 4,
    }

with torch.no_grad():

    for images, poses, focal_lengths, file_names, scene_indices, scores_label in testset_loader:
        images = torch.stack(images).cuda()
        NN, BB, CC, HH, WW = images.size()
        images_in = images.view(NN * BB, CC, HH, WW)
		# print("Example pose content:", poses[0][0])
        print(images_in.shape) 
        scene_indices = torch.cat(scene_indices)
        scores = network(images_in, scene_indices)
        # print(scores.shape)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        # print(scores.shape)
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        print(f'scores: {scores.shape}')
        # keypointimages_folder = f'{output_folder_keypoints}/keypointimages_scene_{n}'
        # os.makedirs(keypointimages_folder, exist_ok=True)
        keypoints_folder = f'{output_folder_keypoints}/keypoints'
        os.makedirs(keypoints_folder, exist_ok=True)
        # one_images = images[n]
        # print(one_images.shape)

        nms_scores = simple_nms(scores, default_config['nms_radius'])
        keypoints = [
            torch.nonzero(s > default_config['keypoint_threshold'])
            for s in nms_scores]
        nms_scores = [s[tuple(k.t())] for s, k in zip(nms_scores, keypoints)]
        keypoints, nms_scores = list(zip(*[
            remove_borders(k, s, default_config['remove_borders'], 480, 640)
            for k, s in zip(keypoints, nms_scores)]))
        if default_config['max_keypoints'] >= 0:
            keypoints, nms_scores = list(zip(*[
                top_k_keypoints(k, s, default_config['max_keypoints'])
                for k, s in zip(keypoints, nms_scores)]))
            
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]
        keypoints = torch.stack(keypoints, dim=0).cpu().numpy()
        print(keypoints.shape)

        keypoints_path = os.path.join(keypoints_folder, f'{counter}.npy')
        np.save(keypoints_path, keypoints)
        counter += 1

print("\n===================================================")
print("\nComplete.")