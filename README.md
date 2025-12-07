# SeqACE_PointHead

This repository contains the implementation of the training code of the pointhead in paper ["Enhancing scene coordinate regression with efficient keypoint detection and sequential information"](https://github.com/sair-lab/SeqACE) by Kuan Xu, Zeyu Jiang, Haozhi Cao, Shenghai Yuan, Chen Wang, and Lihua Xie, published in IEEE Robotics and Automation Letters, 2025.

## Environment Setup
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


## Datasets Preparation
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


## Training
If only RGB images and ground truth poses are available (minimal setup), initialize a network by calling:

```bash
python train.py {output_network.pt} --session experiment_pointhead
```


## Validation pointhead results
1. Use bash to generate all scenes keypoints files in `./results_keypoint/output_keypoints`.
```bash
python test.py {output_network.pt}
```
2. Use `keypoints_visualize.py` to generate keypoints images of output keypoints in origin datasets rgb files to visualize and verify the keypoints training results.


## Citations
If you find our work useful in your research, please consider citing:
```
@article{xu2025enhancing,
  title={Enhancing scene coordinate regression with efficient keypoint detection and sequential information},
  author={Xu, Kuan and Jiang, Zeyu and Cao, Haozhi and Yuan, Shenghai and Wang, Chen and Xie, Lihua},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```

Our code builds on ACE. Please consider citing:

```
@inproceedings{brachmann2023ace,
    title={Accelerated Coordinate Encoding: Learning to Relocalize in Minutes using RGB and Poses},
    author={Brachmann, Eric and Cavallari, Tommaso and Prisacariu, Victor Adrian},
    booktitle={CVPR},
    year={2023},
}
```