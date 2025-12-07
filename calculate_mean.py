import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class MeanDataset(Dataset):
	"""Camera localization dataset.

	Access to image, calibration and ground truth data given a dataset directory.
	"""

	def __init__(self, root_dir):
		pose_dir =  root_dir + '/poses/'


		def sort_key(file):
			return int(os.path.splitext(os.path.basename(file))[0])
		
		self.pose_files = [os.path.join(pose_dir, f) for f in sorted(os.listdir(pose_dir), key=sort_key)]

		self.pose_transform = transforms.Compose([
			transforms.ToTensor()
			])

	def __len__(self):
		return len(self.pose_files)

	def __getitem__(self, index):
		pose = np.loadtxt(self.pose_files[index])
		pose = torch.from_numpy(pose).float()
		return pose
	

def calculate_mean_coordinates(trainset_loader):
    print("Calculating mean scene coordinates for each scene...")

    means = torch.zeros((3))
    counts = 0

    for pose in trainset_loader:

        means += pose[0, 0:3, 3]
        counts += 1     

    mean_coordinates = {}
    formatted_key = f"{scene_name}"
    mean_coordinates[formatted_key] = means / counts


    print("Done calculating mean coordinates for each scene.")
    return mean_coordinates

base_dir = './datasets/7scenes/chess/test/'
scene_name = 'chess'


dataset = MeanDataset(base_dir)
trainset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

print("Found %d training images for the largest scene." % (len(trainset_loader)))

mean = calculate_mean_coordinates(trainset_loader)
torch.save(mean, './mean/mean_chess_test.pth')