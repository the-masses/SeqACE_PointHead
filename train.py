import numpy as np
import random
import torch
import torch.optim as optim
# from torch.cuda.amp import GradScaler, autocast

import time
import argparse
import math
from torch.utils.tensorboard import SummaryWriter

from dataset import MultiCamLocDataset
from network import Network
# from calculate_mean import calculate_mean_coordinates
from loss import superpoint_loss

# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

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


parser.add_argument('--pretrained_encoder', '-pe', type=str, default='./ace_encoder_pretrained.pt',
    help='pretrained encoder model file path to load first ten layers parameters')

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
        multi_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
    )
    
    return trainset_loader


base_dir = './datasets/ScanNet/'

parallel = 1
serial = 1

writer = SummaryWriter('runs/training_experiment_{}'.format(opt.session))
network = Network(num_datasets=serial*parallel)

layers_to_freeze = [
    'conv1', 'conv2', 'conv3', 'conv4',
    'res1_conv1', 'res1_conv2', 'res1_conv3',
    'res2_conv1', 'res2_conv2', 'res2_conv3',
    'res2_skip'
]

for name, param in network.named_parameters():
    if any(layer in name for layer in layers_to_freeze):
        param.requires_grad = False

if opt.pretrained_encoder is not None:
    pretrained_dict = torch.load(opt.pretrained_encoder)
    model_dict = network.state_dict()
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if any(layer in k for layer in layers_to_freeze)}
    
    model_dict.update(pretrained_dict)
    
    network.load_state_dict(model_dict)
    
    print(f"Loaded {opt.pretrained_encoder}.")

network = torch.nn.DataParallel(network)
network = network.cuda()
network.train()

if hasattr(network.module, 'superpoint_head'):
    superpoint_head = network.module.superpoint_head.parameters()
else:
    raise AttributeError("Cannot find 'superpoint_head' attribute in the network module.")


optimizer = optim.Adam(superpoint_head, lr=opt.learningrate)

iteration = 0
epochs = 1000

all_dataset_loaders = []
for n in range(1, serial + 1):
    trainset_loader = load_datasets(base_dir, parallel, n)
    resettable_loader = ResettableLoader(trainset_loader)
    all_dataset_loaders.append(resettable_loader)

loader_lengths = [len(loader) for loader in all_dataset_loaders]
max_loader_idx = loader_lengths.index(max(loader_lengths))
print(f'max_loader_idx: {max_loader_idx}')
max_loader = all_dataset_loaders[max_loader_idx]

for epoch in range(epochs):	
	print("=== Epoch: %d ======================================" % epoch)
	num_batches = len(max_loader)
	for batch_index in range(num_batches):
		max_batch = next(max_loader)
		batch_data = [next(loader) if i != max_loader_idx else max_batch for i, loader in enumerate(all_dataset_loaders)]

		for data in batch_data:
			images, poses, focal_lengths, file_names, scene_indices, scores_label = data
			start_time = time.time()
			images = torch.stack(images).cuda()
			NN, BB, CC, HH, WW = images.size()
			images = images.view(NN * BB, CC, HH, WW)
			scene_indices = torch.cat(scene_indices)
			scores_label = torch.stack(scores_label).cuda()
			scores = network(images, scene_indices) 
			scores_loss = superpoint_loss(scores, scores_label)
			scores_loss.backward()
			writer.add_scalar('Scores loss', scores_loss, iteration)
            
			print('Iteration: %6d, Loss: %.1f, Time: %.2fs' % (iteration, scores_loss, time.time()-start_time), flush=True)

			optimizer.step()
			optimizer.zero_grad()
			iteration = iteration + 1

	print('Saving snapshot of the network to %s.' % opt.network)
	torch.save(network.module.state_dict(), opt.network)

	for loader in all_dataset_loaders:
		loader.reset()
	
writer.close()
print('Done without errors.')
