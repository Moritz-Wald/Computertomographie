import numpy as np

from interpolations import nearest_neighbor_interpolate, bilinear_interpolate


def rotate_theta(theta: float) -> np.ndarray:
    """generate rotation matrix for the angle theta

    returns: (2, 2)  2x2 rotation matrix
    """
    return np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)],
    ], dtype=np.float64)


def rotate_image(image: np.ndarray, angles, method: str = 'nearest'):
    """Rotates an image by the given angles using a chosen interpolation method.

    Arguments:
        image (np.ndarray): an NxN grayscale image to rotate
        angles (np.ndarray): a list of rotation angles to rotate the image with
        method (str): interpolation method ('nearest', 'bilinear')

    Returns:
        np.ndarray: (len(angles)xNxN) array of the rotated images
    """

    if method == 'nearest':
        interpolate = nearest_neighbor_interpolate
    elif method == 'bilinear':
        interpolate = bilinear_interpolate
    else:
        raise ValueError('method must be either "nearest" or "bilinear"')

    height, width = image.shape

    if not height == width:
        raise ValueError('image must be square')

    cx, cy = width / 2, height / 2

    y, x = np.meshgrid(np.arange(height), np.arange(width))
    x = x - cx
    y = y - cy

    angles_rad = np.deg2rad(angles)

    rotated_images = np.zeros((len(angles), height, width), dtype=image.dtype)

    for i, angle in enumerate(angles_rad):
        new_coords = np.matmul(rotate_theta(angle), np.vstack((x.flatten(), y.flatten())))
        new_x = new_coords[0].reshape(height, width) + cx
        new_y = new_coords[1].reshape(height, width) + cy

        new_x = np.clip(new_x, 0, width - 1)
        new_y = np.clip(new_y, 0, height - 1)

        rotated_images[i] = interpolate(image, new_x, new_y)

    return rotated_images


def radon(image: np.ndarray,
          angles: np.ndarray = np.arange(180),
          method: str = 'nearest') -> np.ndarray:
    """Apply a radon transform to an image.

    Arguments:
        image (np.ndarray): an NxN grayscale image to radon transform
        angles (np.ndarray): a list of rotation angles to use in the radon transform
        method (str): interpolation method to use in rotation for discrete image ('nearest', 'bilinear')

    Returns:
        np.ndarray: The sinogram result of the radon transform.
    """
    return rotate_image(image, angles, method).sum(axis=-1).T
