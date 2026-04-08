import numpy as np
import matplotlib.pyplot as plt


def make_rgb(data):
    """
    Construct a true-color RGB image from selected hyperspectral bands.

    Band indices approximate visible Red, Green, and Blue wavelengths
    in the EMIT sensor's spectral configuration:
        - Band 28 ≈ Red
        - Band 17 ≈ Green
        - Band 7  ≈ Blue

    Each channel is normalized independently using 2nd–98th percentile
    stretching to improve contrast and handle outliers.

    Args:
        data (np.ndarray): 3D hyperspectral array of shape (rows, cols, bands).

    Returns:
        np.ndarray: Float32 RGB image of shape (rows, cols, 3), values in [0, 1].
    """
    # Band indices corresponding to approximate R, G, B wavelengths
    r, g, b = 28, 17, 7

    # Stack selected bands into a 3-channel image (rows, cols, 3)
    rgb = np.stack([
        data[:, :, r],
        data[:, :, g],
        data[:, :, b]
    ], axis=2)

    # Clamp any residual negative values to 0 before normalization
    rgb = np.where(rgb < 0, 0, rgb)

    # Normalize each channel independently using percentile stretch
    # This clips extreme outliers and scales values to [0, 1]
    for i in range(3):
        p2 = np.nanpercentile(rgb[:, :, i], 2)
        p98 = np.nanpercentile(rgb[:, :, i], 98)

        rgb[:, :, i] = np.clip(
            (rgb[:, :, i] - p2) / (p98 - p2),
            0, 1
        )

    return rgb


def make_false_color(data):
    """
    Construct a false-color composite image using NIR, Red, and Green bands.

    False-color composites map Near-Infrared (NIR) to the Red channel,
    making vegetation appear in bright red tones — useful for land cover
    and vegetation analysis.

        - Band 100 ≈ NIR  → mapped to Red channel
        - Band 28  ≈ Red  → mapped to Green channel
        - Band 17  ≈ Green → mapped to Blue channel

    Each channel is normalized independently using 2nd–98th percentile
    stretching to improve contrast and handle outliers.

    Args:
        data (np.ndarray): 3D hyperspectral array of shape (rows, cols, bands).

    Returns:
        np.ndarray: Float32 false-color image of shape (rows, cols, 3), values in [0, 1].
    """
    # NIR band highlights vegetation; red/green provide spectral context
    nir, red, green = 100, 28, 17

    # Stack NIR, Red, Green into a 3-channel false-color composite
    fc = np.stack([
        data[:, :, nir],
        data[:, :, red],
        data[:, :, green]
    ], axis=2)

    # Clamp negative values before normalization
    fc = np.where(fc < 0, 0, fc)

    # Percentile-based normalization per channel to stretch contrast
    for i in range(3):
        p2 = np.nanpercentile(fc[:, :, i], 2)
        p98 = np.nanpercentile(fc[:, :, i], 98)

        fc[:, :, i] = np.clip(
            (fc[:, :, i] - p2) / (p98 - p2),
            0, 1
        )

    return fc


def save_images(rgb, fc):
    """
    Save the RGB and false-color images as PNG files to the outputs directory.

    Args:
        rgb (np.ndarray): True-color RGB image of shape (rows, cols, 3).
        fc (np.ndarray):  False-color composite image of shape (rows, cols, 3).
    """
    # Save true-color RGB composite
    plt.imsave("outputs/rgb.png", rgb)

    # Save false-color NIR composite
    plt.imsave("outputs/false_color.png", fc)