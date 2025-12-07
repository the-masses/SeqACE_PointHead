import cv2
import numpy as np
from skimage.transform import warp
from scipy.spatial.transform import Rotation as R


def random_homography_transform(image, focal_length, max_angle=40):
    height, width = image.shape[:2]
    
    K = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ])
    
    angle_rad = np.deg2rad(np.random.uniform(-max_angle, max_angle))
    axis = np.random.rand(3) - 0.5
    rotation = R.from_rotvec(angle_rad * axis).as_matrix()
    
    H = K @ rotation @ np.linalg.inv(K)
    H /= H[2, 2]
    
    warped_image = warp(image, np.linalg.inv(H), output_shape=(height, width))
    
    return warped_image

def add_shade(img, random_state=None, nb_ellipses=10,
              amplitude=[-0.5, 0.5], kernel_size_interval=(150, 250)):
    """ Overlay the image with several shades
    Parameters:
      nb_ellipses: number of shades
      amplitude: tuple containing the illumination bound (between -1 and 0) and the
        shawdow bound (between 0 and 1)
      kernel_size_interval: interval of the kernel used to blur the shades
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    transparency = random_state.uniform(*amplitude)

    min_dim = min(img.shape) / 4
    mask = np.zeros(img.shape[:2], np.uint8)
    for i in range(nb_ellipses):
        ax = int(max(random_state.rand() * min_dim, min_dim / 5))
        ay = int(max(random_state.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = random_state.randint(max_rad, img.shape[1] - max_rad)  # center
        y = random_state.randint(max_rad, img.shape[0] - max_rad)
        angle = random_state.rand() * 90
        cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

    kernel_size = int(kernel_size_interval[0] + random_state.rand() *
                      (kernel_size_interval[1] - kernel_size_interval[0]))
    if (kernel_size % 2) == 0:  # kernel_size has to be odd
        kernel_size += 1
    mask = cv2.GaussianBlur(mask.astype(np.float64), (kernel_size, kernel_size), 0)
    mask = np.expand_dims(mask, axis=-1)
    mask = np.tile(mask, (1, 1, 3))
    shaded = img * (1 - transparency * mask/255.)
    shaded = np.clip(shaded, 0, 255)
    return shaded.astype(np.uint8)


def add_fog(img, random_state=None, max_nb_ellipses=10,
            transparency=0.6, kernel_size_interval=(150, 250)):
    """ Overlay the image with several shades
    Parameters:
      max_nb_ellipses: number max of shades
      transparency: level of transparency of the shades (1 = no shade)
      kernel_size_interval: interval of the kernel used to blur the shades
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    centers = np.empty((0, 2), dtype=np.int32)
    rads = np.empty((0, 1), dtype=np.int32)
    min_dim = min(img.shape) / 4
    shaded_img = img.copy()
    for i in range(max_nb_ellipses):
        ax = int(max(random_state.rand() * min_dim, min_dim / 5))
        ay = int(max(random_state.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = random_state.randint(max_rad, img.shape[1] - max_rad)  # center
        y = random_state.randint(max_rad, img.shape[0] - max_rad)
        new_center = np.array([[x, y]])

        # Check that the ellipsis will not overlap with pre-existing shapes
        diff = centers - new_center
        if np.any(max_rad > (np.sqrt(np.sum(diff * diff, axis=1)) - rads)):
            continue
        centers = np.concatenate([centers, new_center], axis=0)
        rads = np.concatenate([rads, np.array([[max_rad]])], axis=0)

        col = random_state.randint(256)  # color of the shade
        angle = random_state.rand() * 90
        cv2.ellipse(shaded_img, (x, y), (ax, ay), angle, 0, 360, col, -1)
    shaded_img = shaded_img.astype(float)
    kernel_size = int(kernel_size_interval[0] + random_state.rand() *
                      (kernel_size_interval[1] - kernel_size_interval[0]))
    if (kernel_size % 2) == 0:  # kernel_size has to be odd
        kernel_size += 1

    cv2.GaussianBlur(shaded_img, (kernel_size, kernel_size), 0, shaded_img)
    mask = np.where(shaded_img != img)
    shaded_img[mask] = (1 - transparency) * shaded_img[mask] + transparency * img[mask]
    shaded_img = np.clip(shaded_img, 0, 255)
    return shaded_img.astype(np.uint8)


# def motion_blur(img, max_ksize=5):
def motion_blur(img, max_ksize=8):
    # Either vertial, hozirontal or diagonal blur
    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(0, (max_ksize+1)/2)*2 + 1  # make sure is odd

    center = int((ksize-1)/2)
    kernel = np.zeros((ksize, ksize))
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)
    var = ksize * ksize / 16.
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    img = cv2.filter2D(img.astype(np.uint8), -1, kernel)
    return img

# def additive_gaussian_noise(img, random_state=None, std=(0, 10)):
def additive_gaussian_noise(img, random_state=None, std=(0, 15)):
    """ Add gaussian noise to the current image pixel-wise
    Parameters:
      std: the standard deviation of the filter will be between std[0] and std[0]+std[1]
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    sigma = std[0] + random_state.rand() * std[1]
    gaussian_noise = random_state.randn(*img.shape) * sigma
    noisy_img = img + gaussian_noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


# def additive_speckle_noise(img, intensity=1):
def additive_speckle_noise(img, intensity=2):
    """ Add salt and pepper noise to an image
    Parameters:
      intensity: the higher, the more speckles there will be
    """
    noise = np.zeros(img.shape, dtype=np.uint8)
    cv2.randu(noise, 0, 256)
    black = noise < intensity
    white = noise > 255 - intensity
    noisy_img = img.copy()
    noisy_img[white > 0] = 255
    noisy_img[black > 0] = 0
    return noisy_img


# def random_brightness(img, random_state=None, max_change=50):
def random_brightness(img, random_state=None, max_change=80):
    """ Change the brightness of img
    Parameters:
      max_change: max amount of brightness added/subtracted to the image
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    brightness = random_state.randint(-max_change, max_change)
    new_img = img.astype(np.int16) + brightness
    return np.clip(new_img, 0, 255).astype(np.uint8)


# def random_contrast(img, random_state=None, max_change=[0.5, 1.5]):
def random_contrast(img, random_state=None, max_change=[0.5, 2.0]):
    """ Change the contrast of img
    Parameters:
      max_change: the change in contrast will be between 1-max_change and 1+max_change
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    contrast = random_state.uniform(*max_change)
    mean = np.mean(img, axis=(0, 1))
    new_img = np.clip(mean + (img - mean) * contrast, 0, 255)
    return new_img.astype(np.uint8)