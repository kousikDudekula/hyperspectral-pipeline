import numpy as np

def clean_data(data):
    """
    Clean hyperspectral radiance data by replacing invalid values with NaN.

    Negative radiance values are physically meaningless and are treated
    as sensor artifacts or fill values, so they are masked out.

    Args:
        data (np.ndarray): Raw radiance array of any shape, dtype float.

    Returns:
        np.ndarray: Cleaned array of the same shape, with negative values
                    replaced by NaN.
    """

    # Replace any negative radiance values with NaN
    # Negative values are physically invalid (radiance cannot be negative)
    return np.where(data < 0, np.nan, data)