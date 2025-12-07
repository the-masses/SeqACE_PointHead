import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
	'''
	FCN architecture for scene coordiante regression.

	The network predicts a 3d scene coordinates, the output is subsampled by a factor of 8 compared to the input.
	'''

	OUTPUT_SUBSAMPLE = 8

	def __init__(self, mean_coordinates, num_datasets=1):
		'''
		Constructor.
		'''
		super(Network, self).__init__()

		self.conv1a = nn.Conv2d(1, 32, 3, 1, 1)
		self.conv1b = nn.Conv2d(32, 32, 3, 1, 1)
		self.conv2a = nn.Conv2d(32, 64, 3, 2, 1)
		self.conv2b = nn.Conv2d(64, 64, 3, 1, 1)
		self.conv3a = nn.Conv2d(64, 128, 3, 2, 1)
		self.conv3b = nn.Conv2d(128, 128, 3, 1, 1)
		self.conv4a = nn.Conv2d(128, 256, 3, 2, 1)
		self.conv4b = nn.Conv2d(256, 256, 3, 1, 1)

		self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
		self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
		self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

		self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
		self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
		self.res2_conv3 = nn.Conv2d(512, 512, 3, 1, 1)

		self.res2_skip = nn.Conv2d(256, 512, 1, 1, 0)

		self.res3_conv1 = nn.Conv2d(512, 512, 1, 1, 0)
		self.res3_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
		self.res3_conv3 = nn.Conv2d(512, 512, 1, 1, 0)
        
		self.regression_heads = nn.ModuleDict({
                  f"head_{i}": RegressionHead(512) for i in range(num_datasets)
		})

		self.superpoint_head = SuperPointHead()

		init_value = torch.tensor([1.0]).cuda()
		self.logvars = nn.ParameterList([nn.Parameter(init_value.clone()) for _ in range(2)])
        
		# learned scene coordinates relative to a mean coordinate (e.g. center of the scene)
		self.means = mean_coordinates
		# self.temp = 0.1
		# self.temp = nn.Parameter(torch.tensor([temp]).float())
		# self.register_hooks()
	       
	# def register_hooks(self):
	# 	for name, module in self.named_modules():
	# 		module.register_forward_hook(forward_hook)
	# 		module.register_backward_hook(backward_hook)

	def forward(self, inputs, scene_indices):
		'''
		Forward pass.

		inputs -- 5D data tensor (NxBxCxHxW)
		'''
		# if inputs.dim() != 5:
		# 	raise ValueError("Expected input to be a 5D tensor")

		# B, N, C, H, W = inputs.size()
		# inputs = inputs.reshape(B * N, C, H, W)
		x = F.relu(self.conv1a(inputs))
		x = F.relu(self.conv1b(x))
		x = F.relu(self.conv2a(x))
		x = F.relu(self.conv2b(x))
		x = F.relu(self.conv3a(x))
		x = F.relu(self.conv3b(x))
		x = F.relu(self.conv4a(x))
		res = F.relu(self.conv4b(x))
			
		x = F.relu(self.res1_conv1(res))
		x = F.relu(self.res1_conv2(x))
		x = F.relu(self.res1_conv3(x))
			
		res = res + x
			
		x = F.relu(self.res2_conv1(res))
		x = F.relu(self.res2_conv2(x))
		x = F.relu(self.res2_conv3(x))
			
		res = self.res2_skip(res)
				
		res = res + x
			
		x = F.relu(self.res3_conv1(res))
		x = F.relu(self.res3_conv2(x))
		x = F.relu(self.res3_conv3(x))
			
		res = res + x
          
		superpoint_outputs = self.superpoint_head(res)
		scores = superpoint_outputs
		
		scene_outputs = []
		unique_indices = scene_indices.unique()
		for idx in unique_indices:
			scene_mask = (scene_indices == idx)
			# print("scene_mask.shape = {}, res = {}".format(scene_mask.shape, res.shape))
			# print("scene_mask = {}".format(scene_mask))
			scene_specific_features = res[scene_mask].view(-1, res.size(1), res.size(2), res.size(3))
			# print(f"Processing scene index {idx} with data batch size {B}")
			scene_output = self.regression_heads[f"head_{idx.item()}"](scene_specific_features)
			mean = self.means[f"scene{str(idx.item()).zfill(4)}_00"]
			scene_output[:, 0, :, :] += mean[0]
			scene_output[:, 1, :, :] += mean[1]
			scene_output[:, 2, :, :] += mean[2]
			scene_outputs.append(scene_output)
			for idx, tensor in enumerate(scene_outputs):
				print("Shape of tensor {} = {}".format(idx, tensor.shape))		
		
		# 4 dimension output B*N,C,H,W
		scene_outputs = torch.cat(scene_outputs, dim=0)
		print("scene_outputs 1  = {}".format(scene_outputs.shape))
		# print("scores   = {}".format(scores.shape))
            
		return scene_outputs, scores

class RegressionHead(nn.Module):
    def __init__(self, input_channels, output_channels=3, hidden_dim=512):
        super(RegressionHead, self).__init__()
        self.fc_layers = nn.ModuleList([
            nn.Conv2d(input_channels, hidden_dim, 1, 1, 0),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        ])
        self.fc_output = nn.Conv2d(hidden_dim, output_channels, 1, 1, 0)
        self.skip_connection = nn.Conv2d(input_channels, hidden_dim, 1, 1, 0)

    def forward(self, x):
        res = x
        for i, fc_layer in enumerate(self.fc_layers):
            x = F.relu(fc_layer(x))
            if i == 2:
                skip = self.skip_connection(res)
                x = x + skip
        x = self.fc_output(x)

        return x
	
class SuperPointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.convPa = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.convPc = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.convPd = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convPe = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.convPf = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.convPg = nn.Conv2d(128, 65, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Compute keypoints, scores for image """

        # Compute the dense keypoint scores
        x = self.relu(self.convPa(x))
        x = self.relu(self.convPb(x))
        x = self.relu(self.convPc(x))
        x = self.relu(self.convPd(x))
        x = self.relu(self.convPe(x))
        x = self.relu(self.convPf(x))
        scores = self.convPg(x)
        # print('scores.shape', scores.shape)
        # scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        # b, _, h, w = scores.shape
        # scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        # scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        return scores


# def forward_hook(module, input, output):
# 			print(f"Forward hook in {module.__class__.__name__}")
# 			print(f" - Input shape: {tuple(input[0].shape)}")
# 			print(f" - Output shape: {tuple(output.shape)}")

# def backward_hook(module, grad_input, grad_output):
# 		print(f"Backward hook in {module.__class__.__name__}")
# 		print(f" - Grad Input shape: {tuple(g.shape for g in grad_input if g is not None)}")
# 		print(f" - Grad Output shape: {tuple(g.shape for g in grad_output if g is not None)}")    

# mean = torch.load('./mean/mean_2.pth')
# model = Network(mean, num_datasets=2)
# for m in model.modules():
#       print(m)