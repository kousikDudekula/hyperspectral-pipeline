import netCDF4 as nc
import numpy as np

def load_data(path, rows=200, cols=200, bands=150):
    """
    Load a subset of hyperspectral radiance data from a NetCDF4 file.

    Args:
        path (str): Path to the .nc file.
        rows (int): Number of spatial rows to load. Default is 200.
        cols (int): Number of spatial columns to load. Default is 200.
        bands (int): Number of spectral bands to load. Default is 150.

    Returns:
        np.ndarray: 3D array of shape (rows, cols, bands) with float32 values.
    """

    # Open the NetCDF4 dataset in read mode
    ds = nc.Dataset(path)

    # Extract a spatial and spectral subset of the 'radiance' variable
    # Slicing avoids loading the full dataset into memory
    data = ds.variables['radiance'][:rows, :cols, :bands]

    # Convert to float32 NumPy array for downstream processing
    data = np.array(data, dtype=np.float32)

    # Close the dataset to free file handle resources
    ds.close()

    return data