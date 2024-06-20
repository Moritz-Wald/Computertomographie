import numpy as np


def bilinear_interpolate(image, x, y):
    """ THIS WAS TAKEN FROM THE INTERNET"""
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)

    x0 = np.clip(x0, 0, image.shape[1] - 1)
    y0 = np.clip(y0, 0, image.shape[0] - 1)

    x1 = x0 + 1
    y1 = y0 + 1

    x1 = np.clip(x1, 0, image.shape[1] - 1)
    y1 = np.clip(y1, 0, image.shape[0] - 1)

    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]

    denom = (x1 - x0) * (y1 - y0)
    denom[denom == 0] = 1  # Handle division by zero case
    wa = ((x1 - x) * (y1 - y)) / denom
    wb = ((x1 - x) * (y - y0)) / denom
    wc = ((x - x0) * (y1 - y)) / denom
    wd = ((x - x0) * (y - y0)) / denom

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def nearest_neighbor_interpolate(img, x, y):
    """ THIS WAS ADAPTED FROM THE BILINEAR ONE PULLED FROM THE INTERNET ABOVE"""
    x_rounded = np.round(x).astype(int)
    y_rounded = np.round(y).astype(int)

    x_rounded = np.clip(x_rounded, 0, img.shape[1] - 1)
    y_rounded = np.clip(y_rounded, 0, img.shape[0] - 1)

    return img[y_rounded, x_rounded]