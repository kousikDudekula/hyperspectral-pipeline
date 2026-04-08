from sklearn.decomposition import PCA
import numpy as np


def apply_pca(data, n_components=3):
    """
    Apply PCA to a hyperspectral image to reduce spectral dimensionality.

    The 3D image is flattened to 2D (pixels × bands), NaN values are imputed
    with per-band means, PCA is fitted and applied, then the result is reshaped
    back to (rows, cols, n_components).

    NaN handling is done in two passes:
        1. Replace NaNs with per-band column means (preserves spectral structure)
        2. Final np.nan_to_num safety pass for any all-NaN bands (mean would be NaN)

    Args:
        data (np.ndarray): 3D hyperspectral array of shape (rows, cols, bands).
        n_components (int): Number of principal components to retain. Default is 3.

    Returns:
        np.ndarray: PCA-reduced array of shape (rows, cols, n_components).
                    Each channel corresponds to a principal component,
                    ordered by descending explained variance.
    """
    r, c, b = data.shape

    # Flatten spatial dims → (rows*cols, bands) for sklearn compatibility
    reshaped = data.reshape(-1, b)

    # Compute per-band mean, ignoring NaNs, for imputation
    col_mean = np.nanmean(reshaped, axis=0)

    # If an entire band is NaN, nanmean returns NaN — replace with 0
    # to avoid propagating NaNs into PCA
    col_mean = np.where(np.isnan(col_mean), 0, col_mean)

    # Impute NaN pixels with their band's mean value
    # np.take maps each NaN's column index to the corresponding band mean
    inds = np.where(np.isnan(reshaped))
    reshaped[inds] = np.take(col_mean, inds[1])

    # Final safety pass — catches any NaNs missed by mean imputation
    reshaped = np.nan_to_num(reshaped, nan=0.0)

    # Fit PCA on all pixels and project to n_components dimensions
    # Components are ordered by descending explained variance
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(reshaped)

    # Reshape back to spatial layout (rows, cols, n_components)
    return reduced.reshape(r, c, n_components)