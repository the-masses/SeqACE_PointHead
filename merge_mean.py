import torch

def merge_and_sort_pth_files(file_paths, output_path):
    merged_data = {}
    # Loop through all file paths and merge their contents
    for path in file_paths:
        data = torch.load(path)
        merged_data.update(data)  # Merge data into the main dictionary

    # Sort the dictionary by keys (scene numbers)
    sorted_data = {key: merged_data[key] for key in sorted(merged_data)}

    # Save the sorted dictionary to a new .pth file
    torch.save(sorted_data, output_path)

    return sorted_data

# List of file paths
file_paths = [f'./mean/mean_{i}.pth' for i in range(10)]

# Output file path
output_path = './mean/mean.pth'

# Call the function and get the merged, sorted data
merged_sorted_data = merge_and_sort_pth_files(file_paths, output_path)

# Print the sorted data
print(merged_sorted_data)
