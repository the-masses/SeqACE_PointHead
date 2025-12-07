import torch
import torch.optim as optim
# from torch.cuda.amp import GradScaler, autocast

import time
import argparse
import math
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle

from dataset import MultiCamLocDataset
from network import Network
# from calculate_mean import calculate_mean_coordinates
from loss import compute_loss

parser = argparse.ArgumentParser(
	description='Initialize a scene coordinate regression network.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('network', help='output file name for the network')

parser.add_argument('--learningrate', '-lr', type=float, default=0.0001, 
	help='learning rate')

parser.add_argument('--iterations', '-iter', type=int, default=1000000,
	help='number of training iterations, i.e. numer of model updates')

parser.add_argument('--inittolerance', '-itol', type=float, default=0.1, 
	help='switch to reprojection error optimization when predicted scene coordinate is within this tolerance threshold to the ground truth scene coordinate, in meters')

parser.add_argument('--mindepth', '-mind', type=float, default=0.1, 
	help='enforce  predicted scene coordinates to be this far in front of the camera plane, in meters')

parser.add_argument('--maxdepth', '-maxd', type=float, default=1000, 
	help='enforce that scene coordinates are at most this far in front of the camera plane, in meters')

parser.add_argument('--targetdepth', '-td', type=float, default=10, 
	help='if ground truth scene coordinates are unknown, use a proxy scene coordinate on the pixel ray with this distance from the camera, in meters')

parser.add_argument('--softclamp', '-sc', type=float, default=100, 
	help='robust square root loss after this threshold, in pixels')

parser.add_argument('--hardclamp', '-hc', type=float, default=1000, 
	help='clamp loss with this threshold, in pixels')

parser.add_argument('--session', '-sid', default='',
	help='custom session name appended to output files, useful to separate different runs of a script')

parser.add_argument('--save_start', '-ss', type=int, default=100, 
	help='Start saving after this epoch')

parser.add_argument('--save_every', '-se', type=int, default=50, 
	help='Save every m epochs after start')

opt = parser.parse_args()



class ResettableLoader:
    def __init__(self, loader):
        self.original_loader = loader
        self.reset()

    def reset(self):
        self.iterator = iter(self.original_loader)

    def restart(self):
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.reset()
            data = next(self.iterator)
        return data

    def __len__(self):
        return len(self.original_loader)


def load_datasets(base_dir, parallel, n):
    start_idx = (n - 1) * parallel
    end_idx = n * parallel - 1
    
    dataset_configs = [
        (f'{base_dir}scene{str(i).zfill(4)}_00', i)
        for i in range(start_idx, end_idx + 1)
    ]
    
    multi_dataset = MultiCamLocDataset(dataset_configs)
    
    trainset_loader = torch.utils.data.DataLoader(
        multi_dataset, batch_size=6, shuffle=True, num_workers=6)
    
    return trainset_loader


base_dir = './datasets/scans/small/'

# Parallel load datasets number on CUDA (Parallel must equal to N times number of CUDA. If you have 3 GPUs, parallel can be 3, 6, 9, ....., 3*N)
parallel = 2
# Serial load datasets number on CPU
serial = 2

# Tensorboard
writer = SummaryWriter('runs/training_experiment_{}'.format(opt.session))

# mean = calculate_mean_coordinates(trainset_loader)
# torch.save(mean, './mean/mean_20.pth')
mean = torch.load('./mean/mean.pth')

# print("Mean coordinates dictionary:", mean)
# print("Keys in mean dictionary:", mean.keys())
# for key, value in mean.items():
#     print(f"Scene: {key}, Mean coordinates: {value.numpy()}")

# create network
network = Network(mean, num_datasets=serial*parallel)
pretrained_model_path = './grediant.pt'
try:
	network.load_state_dict(torch.load(pretrained_model_path))
	print("Loaded pretrained model from", pretrained_model_path)
except Exception as e:
	print("Error loading pretrained model:", e)


network = torch.nn.DataParallel(network)
network = network.cuda()
network.train()

# optimizer = optim.Adam(network.parameters(), lr=opt.learningrate)
# scaler = GradScaler()

optimizer = optim.Adam(network.parameters(), lr=opt.learningrate)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400], gamma=0.1)

# params_regression_heads = list(network.module.regression_heads.parameters())
# params_superpoint_head = list(network.module.superpoint_head.parameters())
# params_backbone = []
# for name, param in network.named_parameters():
#     if "conv" in name or "res" in name:
#         params_backbone.append(param)
# optimizer_regression_heads = optim.Adam(params_regression_heads, lr=0.001)
# optimizer_superpoint_head = optim.Adam(params_superpoint_head, lr=0.001)
# scheduler_superpoint_head = torch.optim.lr_scheduler.MultiStepLR(optimizer_superpoint_head, milestones=[100, 150], gamma=0.1)
# optimizer_backbone = optim.Adam(params_backbone, lr=0.0001)
# scheduler_backbone = torch.optim.lr_scheduler.MultiStepLR(optimizer_backbone, milestones=[100, 150], gamma=0.1)

iteration = 0
epochs = 1000
# Weight parameters of scores_loss
# initial_weight = 40
# decay_rate = 0.996
all_dataset_loaders = []
for n in range(1, serial + 1):
    trainset_loader = load_datasets(base_dir, parallel, n)
    resettable_loader = ResettableLoader(trainset_loader)
    all_dataset_loaders.append(resettable_loader)

loader_lengths = [len(loader) for loader in all_dataset_loaders]
max_loader_idx = loader_lengths.index(max(loader_lengths))
print(max_loader_idx)
max_loader = all_dataset_loaders[max_loader_idx]
# cycle_loaders = [cycle(loader) if i != max_loader_idx else loader for i, loader in enumerate(all_dataset_loaders)]

for epoch in range(epochs):	
	print("=== Epoch: %d ======================================" % epoch)
	num_batches = len(max_loader)
	for batch_index in range(num_batches):
		max_batch = next(max_loader)
		batch_data = [next(loader) if i != max_loader_idx else max_batch for i, loader in enumerate(all_dataset_loaders)]

		for data in batch_data:
			images, poses, coords, focal_lengths, file_names, scene_indices, scores_label = data
			# print(focal_lengths)
			# print(type(focal_lengths))
			# print([type(fl) for fl in focal_lengths])



			start_time = time.time()

			images = torch.stack(images).cuda()
			# change to 4 dimension for network input
			NN, BB, CC, HH, WW = images.size()
			images = images.view(NN * BB, CC, HH, WW)

			# print(type(images))
			# print([img.shape for img in images])
			# print(images.shape)

			scores_label = torch.stack(scores_label).cuda()
			# print(type(scores_label))
			# print([img.shape for img in scores_label])

			scene_indices = torch.cat(scene_indices)
			print("Loaded scene indices:", scene_indices)

			# with autocast():
			# print("scene_indices = {}".format(scene_indices))

			scene_coords, scores = network(images, scene_indices) 
			# scene_coords_time = time.time() - start_time
			# print('Scene_coords_time: %f', scene_coords_time)
			# print('Scene.shape:%f',scene_coords.shape)
			# print('Scores.shape:%f', scores.shape)
			# print("network   = 1")

			# change network output of 3D coordinates into 5 dimension N,B,C,H,W
			C, H, W = scene_coords.shape[1], scene_coords.shape[2], scene_coords.shape[3]
			scene_coords = scene_coords.reshape(NN, BB, C, H, W)
			N, B, _, _, _ = scene_coords.shape
			# scene_coords_shape_time = time.time() - start_time
			# print('Scene_coords_shpe_time: %f', scene_coords_shape_time)
			# create camera calibartion matrix
			# focal_lengths = torch.stack(focal_lengths).cuda()
			# focal_lengths_tensor_time = time.time() - start_time
			# print('Focal_length_tensor_time: %f', focal_lengths_tensor_time)
			# print('Focal_length:%f',focal_lengths_tensor.shape)
			cam_mats = []
			cx = (images.size(3) / 2)
			cy = (images.size(2) / 2)
			for i in range(N):
				float_focal_lengths = focal_lengths[i][0].item()
				cam_mat = torch.eye(3).cuda()
				cam_mat[0, 0] = float_focal_lengths
				cam_mat[1, 1] = float_focal_lengths
				cam_mat[0, 2] = cx
				cam_mat[1, 2] = cy
				cam_mats.append(cam_mat)
			cam_mats = torch.stack(cam_mats)
			# print(cam_mats.shape)
			# cam_mats_time = time.time() - start_time
			# print('Cam_mats_time: %f', cam_mats_time)

			# initialize ground truth grid
			pixel_grid = torch.zeros((N, B, 2, 
				math.ceil(5000 / Network.OUTPUT_SUBSAMPLE),		# 5000px is max limit of image size, increase if needed
				math.ceil(5000 / Network.OUTPUT_SUBSAMPLE)))
					
			# create ground truth grid
			num_scenes, batch_size, _, grid_height, grid_width = scene_coords.shape

			x_coords = torch.arange(grid_width) * Network.OUTPUT_SUBSAMPLE + Network.OUTPUT_SUBSAMPLE / 2
			y_coords = torch.arange(grid_height) * Network.OUTPUT_SUBSAMPLE + Network.OUTPUT_SUBSAMPLE / 2

			pixel_grid = torch.zeros(num_scenes, batch_size, 2, grid_height, grid_width).cuda()
			pixel_grid[:, :, 0, :, :] = x_coords[None, None, None, :]
			pixel_grid[:, :, 1, :, :] = y_coords[None, None, :, None]

			# Define a weight to control scores_loss training balance
			# weight = initial_weight * (decay_rate ** epoch)
			logvars = network.module.logvars if isinstance(network, torch.nn.DataParallel) else network.logvars

			# pixel_grid_time = time.time() - start_time
			# print('Pixel_grid_time: %f', pixel_grid_time)

			avg_loss, scores_loss, both_loss, avg_num_valid_sc, logvars, precisions = compute_loss(scene_coords, poses, cam_mats, pixel_grid, scores, scores_label, opt, logvars)
			# scores_loss = compute_loss(scene_coords, poses, cam_mats, pixel_grid, scores, scores_label, opt, Network.modules.logvars)

			# optimizer.zero_grad()	
			# loss_backward_time = time.time() - start_time
			# print('Loss_backward_time: %f', loss_backward_time)
			both_loss.backward()			# calculate gradients (pytorch autograd)
			# scores_loss.backward()
			writer.add_scalar('Training loss', both_loss, iteration)
			writer.add_scalar('Valid_sc', avg_num_valid_sc, iteration)
			writer.add_scalar('ACE loss', avg_loss, iteration)
			writer.add_scalar('Scores loss', scores_loss, iteration)
			writer.add_scalar('ACE -log(Parameter)', precisions[0], iteration)
			writer.add_scalar('Scores -log(Parameter)', precisions[1], iteration)
			writer.add_scalar('ACE Parameter', logvars[0], iteration)
			writer.add_scalar('Scores Parameter', logvars[1], iteration)
			# writer.add_scalar('Weight', weight, iteration)
			writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], iteration)

			print('Iteration: %6d, Loss: %.1f, ACE loss: %.1f, Valid: %.1f%%, Scores loss: %.3f, Time: %.2fs' % (iteration, both_loss, avg_loss, avg_num_valid_sc*100, scores_loss, time.time()-start_time), flush=True)
			# print('Iteration: %6d, Loss: %.1f, Time: %.2fs' % (iteration, scores_loss, time.time()-start_time), flush=True)

			# optimizer_regression_heads.step()		# update all model parameters
			# optimizer_regression_heads.zero_grad()
			# optimizer_superpoint_head.step()
			# optimizer_superpoint_head.zero_grad()
			# optimizer_backbone.step()
			# optimizer_backbone.zero_grad()
			
			# scheduler_superpoint_head.step()
			# scheduler_backbone.step()
			# scheduler.step()
		optimizer.step()
		optimizer.zero_grad()
		iteration = iteration + 1

	if epoch >= opt.save_start and (epoch - opt.save_start) % opt.save_every == 0:
		base_name = opt.network.split('.pt')[0]
		save_path = f"{base_name}_{epoch}.pt"
		torch.save(network.module.state_dict(), save_path)
		print(f"Model saved to {save_path}")
	print('Saving snapshot of the network to %s.' % opt.network)
	torch.save(network.module.state_dict(), opt.network)
	
writer.close()
print('Done without errors.')
