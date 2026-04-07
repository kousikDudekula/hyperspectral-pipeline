from sklearn.decomposition import PCA
import numpy as np

def apply_pca(data, n_components=3):
    r, c, b = data.shape
    
    reshaped = data.reshape(-1, b)

    # Mean per band
    col_mean = np.nanmean(reshaped, axis=0)

    # Handle all-NaN columns
    col_mean = np.where(np.isnan(col_mean), 0, col_mean)

    # Replace NaNs
    inds = np.where(np.isnan(reshaped))
    reshaped[inds] = np.take(col_mean, inds[1])

    # Final safety
    reshaped = np.nan_to_num(reshaped, nan=0.0)

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(reshaped)

    return reduced.reshape(r, c, n_components)