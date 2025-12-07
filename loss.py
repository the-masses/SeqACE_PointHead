import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def compute_loss(scene_coords, poses, cam_mat, pixel_grid, scores, scores_label, opt, logvars):

	# start_time = time.time()
	total_loss = 0
	total_num_valid_sc = 0
	
    # scene_coords (N, B, C, H, W)
	num_scenes = scene_coords.size(0)
	for i in range(num_scenes):
		single_scene_coords = scene_coords[i]
		# print("single_scene_coords:", single_scene_coords.shape)
		single_pose = poses[i]
		# print("single_pose", single_pose.shape)
		single_pixel_grid = pixel_grid[i]
		# print("single_pixel_grid", single_pixel_grid.shape)
		single_cam_mat = cam_mat[i]
		# print("single_cam_mat", single_cam_mat.shape)
		
		loss, num_valid_sc = compute_single_scene_loss(single_scene_coords, single_pose, single_cam_mat, single_pixel_grid, opt)
		total_loss += loss
		total_num_valid_sc += num_valid_sc

		avg_loss = total_loss / num_scenes
		avg_num_valid_sc = total_num_valid_sc / num_scenes
	
	# loss_time = time.time() - start_time
	# print('Loss_time: %f', loss_time)
	scores_loss = superpoint_loss(scores, scores_label)

	precisions = [torch.exp(-v) for v in logvars]
	both_loss = avg_loss*precisions[0] + logvars[0]
	both_loss += 2*scores_loss*precisions[1] + logvars[1]

	# both_loss = avg_loss + scores_loss * weight
		
	return avg_loss, scores_loss, both_loss, avg_num_valid_sc, logvars, precisions
	# return scores_loss


# calculate loss dependant on the mode
def compute_single_scene_loss(scene_coords, poses, cam_mat, pixel_grid, opt):
	# === RGB mode, optmize a variant of the reprojection error ===================

	# crop ground truth pixel positions to prediction size
	pixel_grid_crop = pixel_grid[:,:,0:scene_coords.size(2),0:scene_coords.size(3)].clone()
	# print("pixel_grid_crop:", pixel_grid_crop.shape)
	# print("pixel_grid_crop:", pixel_grid_crop[:1])
	# pixel_grid_crop = pixel_grid_crop.view(2, -1)
	pixel_grid_crop = pixel_grid_crop.view(pixel_grid_crop.size(0), pixel_grid_crop.size(1), -1)
	# print(pixel_grid_crop.shape)
	# print(pixel_grid_crop[:1])
	# make 3D points homogeneous
	ones = torch.ones((scene_coords.size(0), 1, scene_coords.size(2), scene_coords.size(3)))
	# print("ones:", ones.shape)
	ones = ones.cuda()

	scene_coords = torch.cat((scene_coords, ones), 1)
	scene_coords = scene_coords.view(scene_coords.size(0), scene_coords.size(1), -1)
	# print('scene_coords:', scene_coords[:1])
	# print('scene_coords:', scene_coords.shape)

	# prepare pose for projection operation
	poses_inv = torch.inverse(poses)
	poses = poses_inv[:, :3, :]
	# print('poses:', poses.shape)
	# print('poses:', poses[:1])
	poses = poses.cuda()

	# scene coordinates to camera coordinate 
	camera_coords = torch.bmm(poses, scene_coords)
	# print('camera_coords:', camera_coords[:1])
	# print('camera_coords:', camera_coords.shape)

	# re-project predicted scene coordinates
	cam_mat = cam_mat.repeat(scene_coords.size(0), 1, 1)
	# print('cam_mat', cam_mat[:1])
	# print('cam_mat', cam_mat.shape)
	reprojection_error = torch.bmm(cam_mat, camera_coords)
	# print('reprojection_error1', reprojection_error[:1])
	# print('reprojection_error1', reprojection_error.shape)
	reprojection_error[:, 2, :].clamp_(min=opt.mindepth) # avoid division by zero
	x = reprojection_error[:, 0, :] / reprojection_error[:, 2, :]
	y = reprojection_error[:, 1, :] / reprojection_error[:, 2, :]
	reprojection_error = torch.stack((x, y), dim=1)
	# print('reprojection_error1', reprojection_error[:1])
	# print('reprojection_error1', reprojection_error.shape)

	# print("Scene coords size:", scene_coords.shape)
	# print("Reprojection error size:", reprojection_error.shape)
	# print("Pixel grid crop size:", pixel_grid_crop.shape)

	reprojection_error = reprojection_error - pixel_grid_crop
	reprojection_error = reprojection_error.permute(1, 0, 2).reshape(2, -1)
	reprojection_error = torch.norm(reprojection_error, p=2, dim=0)
	# print('reprojection_error2', reprojection_error.shape)
	# print('reprojection_error2', reprojection_error[:10])

	camera_coords_flat = camera_coords.permute(1, 0, 2).reshape(3, -1)
	# print('camera_coords_flat shape:', camera_coords_flat.shape)

	# check predicted scene coordinate for various constraints
	invalid_min_depth = camera_coords_flat[2] < opt.mindepth # behind or too close to camera plane
	# print("camera_coords_flat[2] = {}".format(camera_coords_flat[2]))
	# print("reprojection_error = {}".format(reprojection_error))

	# print('invalid_min_depth', invalid_min_depth.shape)
	# print('camera_coords[2]', camera_coords.shape)
	invalid_repro = reprojection_error > opt.hardclamp # check for very large reprojection errors

	# no ground truth scene coordinates available, enforce max distance of predicted coordinates
	invalid_max_depth = camera_coords_flat[2] > opt.maxdepth
	# print('invalid_max_depth', invalid_max_depth.shape)
	# print('camera_coords[2]', camera_coords.shape)

	# combine all constraints
	valid_scene_coordinates = (invalid_min_depth + invalid_max_depth + invalid_repro) == 0
	# print('valid scene coordinates', valid_scene_coordinates)
	# print('valid scene coordinates', valid_scene_coordinates.shape)
	# print("invalid_min_depth = {}".format(invalid_min_depth))
	# print("invalid_max_depth = {}".format(invalid_max_depth))
	# print("invalid_repro = {}".format(invalid_repro))

	num_valid_sc = int(valid_scene_coordinates.sum())
	# print('num_valid_sc', num_valid_sc)

	# assemble loss
	loss = 0
	scene_coords_flat = scene_coords.permute(1, 0, 2).reshape(4,-1)
	# print('scene_coords_flat shape:', scene_coords_flat.shape)
	# print('scene_coords_flat shape:', scene_coords_flat.size(1))
				
	if num_valid_sc > 0:

		# reprojection error for all valid scene coordinates
		reprojection_error = reprojection_error[valid_scene_coordinates]
		# print(reprojection_error)
		# print(reprojection_error.shape)
		# calculate soft clamped l1 loss of reprojection error
		loss_l1= reprojection_error[reprojection_error <= opt.softclamp]
		loss_sqrt = reprojection_error[reprojection_error > opt.softclamp]
		loss_sqrt = torch.sqrt(opt.softclamp*loss_sqrt)

		loss += (loss_l1.sum() + loss_sqrt.sum())

	if num_valid_sc < scene_coords_flat.size(1):

		invalid_scene_coordinates = (valid_scene_coordinates == 0) 

		# generate proxy coordinate targets with constant depth assumption
		target_camera_coords = pixel_grid_crop.clone()
		# print("target camera coords:", target_camera_coords.shape)
		# print('cam_mat', cam_mat.shape)
		target_camera_coords[:, 0, :] -= cam_mat[:, 0, 2].unsqueeze(1)
		target_camera_coords[:, 1, :] -= cam_mat[:, 1, 2].unsqueeze(1)
		target_camera_coords[:, 0, :] *= opt.targetdepth
		target_camera_coords[:, 1, :] *= opt.targetdepth
		target_camera_coords[:, 0, :] /= cam_mat[:, 0, 0].unsqueeze(1)
		target_camera_coords[:, 1, :] /= cam_mat[:, 1, 1].unsqueeze(1)
		# make homogeneous
		ones = torch.ones_like(target_camera_coords[:, :1, :])
		target_camera_coords = torch.cat([target_camera_coords, ones], dim=1)
		# print('target camera coords', target_camera_coords[:1])
		# print('target camera coords', target_camera_coords.shape)

		target_camera_coords_flat = target_camera_coords.permute(1, 0, 2).reshape(3, -1)
		# print('target camera coords_flat', target_camera_coords_flat)
		# print('target camera coords_flat', target_camera_coords_flat.shape)
		camera_coords_invalid = camera_coords_flat[:, invalid_scene_coordinates]
		target_camera_coords_invalid = target_camera_coords_flat[:, invalid_scene_coordinates]
		# print('camera coords invalid', camera_coords_invalid)
		# print('camera coords invalid', camera_coords_invalid.shape)
		# print('target coords invalid', target_camera_coords_invalid)
		# print('target coords invalid', target_camera_coords_invalid.shape)

		# distance 
		# loss += torch.abs(camera_coords[:,invalid_scene_coordinates] - target_camera_coords[:, invalid_scene_coordinates]).sum()
		loss += torch.abs(camera_coords_invalid - target_camera_coords_invalid).sum()
		# print('loss', loss.shape)

	loss /= scene_coords_flat.size(1)
	num_valid_sc /= scene_coords_flat.size(1)
	
	return loss, num_valid_sc

def superpoint_loss(scores, scores_label):

	# print("Scores_label shape:", scores_label.shape)
	scores_label = scores_label.view(-1, scores_label.size(2), scores_label.size(3), scores_label.size(4))
	# print("Reshaped scores_label shape:", scores_label.shape)
	# scores = scores.permute(0, 2, 3, 1)
	# print("Reshaped scores shape:", scores.shape)
	# print("Reshaped scores_label shape:", scores_label.shape)
	# scores = scores / 0.01

	loss_function = nn.CrossEntropyLoss()
	# loss_function = nn.MSELoss()
	scores_loss = loss_function(scores, scores_label)

	return scores_loss


