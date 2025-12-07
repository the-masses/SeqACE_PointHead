
# 1. Environment Setup
The repository contains an `environment.yml` for the use with Conda:

```bash
conda env create -f environment.yml
conda activate ace
```

You compile and install the C++ extension by executing:

```bash
cd dsacstar
python setup.py install
```


# 2. Datasets Preparation
First, download the ScanNet dataset from [ScanNet](http://www.scan-net.org/). After downloading, extract the dataset, we need to  and use the provided `./superpoint/get_scores.py` script to generatrte dense heatmap as label.

Then, preprocess the data into the required format:

```bash
datasets/ScanNet/
├── scene0000_00/
│   ├── calibration/
│   │   ├── 0.txt
│   │   ├── 1.txt
│   │   └── ...
│   │       # Camera focal lengths parameter
│   │
│   ├── dense_scores/
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   └── ...
│   │       # Ground-truth scoremap extracted from SuperPoint network
│   │       # Shape: [1296, 968, 65]
│   │
│   ├── poses/
│   │   ├── 0.txt
│   │   ├── 1.txt
│   │   └── ...
│   │       # Camera pose (4×4 SE3 matrix) for each frame
│   │
│   ├── rgb/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   │       # RGB input images (resolution 1296 × 968)
│   │
│   └── ... (optional additional metadata)
│
├── scene0001_00/
│   ├── calibration/
│   ├── dense_scores/
│   ├── poses/
│   ├── rgb/
│   └── ...
│
├── ...

```

Finally, we need to calculate mean coordinate of datasets by using `calculate_mean.py` to get the key and value of mean coordinate. We already calculate 100 scenes mean coordinate of ScanNet and put them into `./mean/`.


# 3. Training
If only RGB images and ground truth poses are available (minimal setup), initialize a network by calling:

```bash
python train_init.py {output_network.pt} --session experiment_pointhead
```


# 4. Validation pointhead results
1. Use bash to generate all scenes keypoints files in `./results_keypoint/output_keypoints`.
```bash
python test_keypoint.py {output_network.pt}
```
2. Use `keypoints_visualize.py` to generate keypoints images of output keypoints in origin datasets rgb files to visualize and verify the keypoints training results.