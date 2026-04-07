from scipy.ndimage import gaussian_filter
import numpy as np

def denoise(data, sigma=1):
    out = np.zeros_like(data)
    
    for i in range(data.shape[2]):
        out[:, :, i] = gaussian_filter(data[:, :, i], sigma)
    
    return out


def calculate_snr(original, denoised):
    signal = np.nanmean(original)
    noise = np.nanstd(original - denoised)
    
    if noise == 0:
        return 0
    
    return round(signal / noise, 2)