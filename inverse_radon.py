from itertools import product

import numpy as np
from numpy.fft import fft, ifft, fftfreq

from radon import rotate_theta


def iradon(sinogram: np.ndarray,
           angles: np.ndarray = np.arange(180),
           filter_type: str = 'ramp') -> np.ndarray:
    """Apply an inverse radon transform to a sinogram.

    Arguments:
        sinogram (np.ndarray): an TxN grayscale image to inverse radon transform.
            T are the angles in degrees and N the side length of the square
            picture to reconstruct.
        angles (np.ndarray): a list of rotation angles to use in the radon transform
        filter_type (str): name of the filter type to use in the inverse radon
            transform to filter high frequencies

    Returns:
        np.ndarray: The reconstruction result from the inverse radon transform
            on the data given by the sinogram.
    """
    N = sinogram.shape[0]
    rec_image = np.zeros(N * N, dtype=sinogram.dtype)
    cen = N / 2

    angles_rad = np.deg2rad(angles)

    coords = np.fromiter(product(range(N), range(N)), dtype=(int, 2)) - cen
    x_rots = rotate_theta(-angles_rad)[0]
    all_rotated_xs = np.matmul(coords, x_rots) + cen

    freqs = fftfreq(N).reshape(-1, 1)
    if filter_type == 'ramp':
        filter = 2 * np.abs(freqs)
    else:
        raise NotImplementedError('The chose filter type is not implemented.')

    projection = fft(sinogram, axis=0) * filter
    radon_filtered = np.real(ifft(projection, axis=0))

    for proj, ts in zip(radon_filtered.T, all_rotated_xs.T):
        rec_image += np.interp(ts, xp=np.arange(N), fp=proj, left=0, right=0)

    rec_image = rec_image.reshape((N, N), order='F')
    return rec_image * np.pi / (2 * len(angles))
