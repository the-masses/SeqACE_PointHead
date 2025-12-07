import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
	'''
	FCN architecture for scene coordiante regression.

	The network predicts a 3d scene coordinates, the output is subsampled by a factor of 8 compared to the input.
	'''

	OUTPUT_SUBSAMPLE = 8

	def __init__(self, num_datasets=1):
		'''
		Constructor.
		'''
		super(Network, self).__init__()

		self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
		self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
		self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

		self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
		self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
		self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

		self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
		self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
		self.res2_conv3 = nn.Conv2d(512, 512, 3, 1, 1)

		self.res2_skip = nn.Conv2d(256, 512, 1, 1, 0)
        

		self.superpoint_head = SuperPointHead()


	def forward(self, inputs, scene_indices):
		'''
		Forward pass.

		inputs -- 5D data tensor (NxBxCxHxW)
		'''

		x = F.relu(self.conv1(inputs))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		res = F.relu(self.conv4(x))
            
		x = F.relu(self.res1_conv1(res))
		x = F.relu(self.res1_conv2(x))
		x = F.relu(self.res1_conv3(x))

		res = res + x

		x = F.relu(self.res2_conv1(res))
		x = F.relu(self.res2_conv2(x))
		x = F.relu(self.res2_conv3(x))
			
		res = self.res2_skip(res)
		res = res + x

		superpoint_outputs = self.superpoint_head(res)
		scores = superpoint_outputs
            
		return scores


class SuperPointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.convPa = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.convPc = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.convPd = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convPe = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # self.convPf = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.convPg = nn.Conv2d(128, 65, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Compute keypoints, scores for image """

        # Compute the dense keypoint scores
        x = self.relu(self.convPa(x))
        x = self.relu(self.convPb(x))
        # x = self.relu(self.convPc(x))
        x = self.relu(self.convPd(x))
        x = self.relu(self.convPe(x))
        # x = self.relu(self.convPf(x))
        scores = self.convPg(x)
        # print('scores.shape', scores.shape)
        # scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        # b, _, h, w = scores.shape
        # scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        # scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        return scores

