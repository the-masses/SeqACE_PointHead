import torch

def calculate_mean_coordinates(trainset_loader):
    """
    Calculate the mean of coordinates for each dataset, skipping duplicates.

    Parameters:
        trainset_loader: DataLoader for the training dataset.
        use_init: Boolean flag to determine the method of calculating mean.
    
    Returns:
        means: A dictionary containing the mean of coordinates for each dataset.
    """
    print("Calculating mean scene coordinates for each scene...")

    means = {}
    counts = {}
    processed_files = set()  # Set to store processed file names

    for images, poses, focal_lengths, file_names, scene_indices in trainset_loader:
        for i, scene_idx in enumerate(scene_indices):
            # Skip if file has been processed already
            if file_names[i] in processed_files:
                continue
            processed_files.add(file_names[i])  # Mark this file as processed

            if scene_idx not in means:
                means[scene_idx] = torch.zeros((3))
                counts[scene_idx] = 0

            current_pose = poses[i]
            if torch.isfinite(current_pose[0, 0:3, 3]).all():
                means[scene_idx] += current_pose[0, 0:3, 3]
                counts[scene_idx] += 1

    mean_coordinates = {}
    for scene_idx in means:
        formatted_key = f"scene{str(scene_idx[0].item()).zfill(4)}_00"
        if counts[scene_idx] > 0:
            mean_coordinates[formatted_key] = means[scene_idx] / counts[scene_idx]
        else:
            print(f"Warning: No valid coordinates found for scene {scene_idx}. Mean is set to NaN.")
            means[formatted_key][:] = float('nan')


    print("Done calculating mean coordinates for each scene.")
    return mean_coordinates
