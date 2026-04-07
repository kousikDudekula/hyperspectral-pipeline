import netCDF4 as nc
import numpy as np

def load_data(path, rows=200, cols=200, bands=150):
    ds = nc.Dataset(path)
    data = ds.variables['radiance'][:rows, :cols, :bands]
    data = np.array(data, dtype=np.float32)
    ds.close()
    return data