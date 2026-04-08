from scipy.ndimage import gaussian_filter
import numpy as np

def denoise(data, sigma=1):
    """
    Apply Gaussian smoothing to each spectral band of a hyperspectral image.

    Gaussian filtering reduces high-frequency noise while preserving
    spatial structure. Each band is filtered independently to avoid
    cross-band interference.

    Args:
        data (np.ndarray): 3D hyperspectral array of shape (rows, cols, bands).
        sigma (float): Standard deviation of the Gaussian kernel.
                       Higher values = stronger smoothing. Default is 1.

    Returns:
        np.ndarray: Denoised array of the same shape as input.
    """

    # Allocate output array with the same shape and dtype as input
    out = np.zeros_like(data)

    # Process each spectral band independently
    # Gaussian filter is applied spatially (2D) per band, not across bands
    for i in range(data.shape[2]):
        out[:, :, i] = gaussian_filter(data[:, :, i], sigma)

    return out


def calculate_snr(original, denoised):
    """
    Calculate the Signal-to-Noise Ratio (SNR) of the denoising process.

    SNR = mean(signal) / std(noise)
    where noise is estimated as the difference between original and denoised data.

    Args:
        original (np.ndarray): Raw radiance array before denoising.
        denoised (np.ndarray): Smoothed array after Gaussian filtering.

    Returns:
        float: Rounded SNR value. Returns 0 if noise standard deviation is zero
               (i.e., original and denoised are identical).
    """

    # Estimate signal as the mean radiance across all valid (non-NaN) pixels
    signal = np.nanmean(original)

    # Estimate noise as the spread of the residual (original minus denoised)
    noise = np.nanstd(original - denoised)

    # Avoid division by zero if denoising had no effect
    if noise == 0:
        return 0

    return round(signal / noise, 2)