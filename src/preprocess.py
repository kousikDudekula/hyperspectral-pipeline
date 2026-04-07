import numpy as np

def clean_data(data):
    return np.where(data < 0, np.nan, data)