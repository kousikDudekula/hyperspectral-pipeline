import numpy as np
import matplotlib.pyplot as plt

def make_rgb(data):
    r, g, b = 28, 17, 7
    
    rgb = np.stack([
        data[:, :, r],
        data[:, :, g],
        data[:, :, b]
    ], axis=2)
    
    rgb = np.where(rgb < 0, 0, rgb)
    
    for i in range(3):
        p2 = np.nanpercentile(rgb[:, :, i], 2)
        p98 = np.nanpercentile(rgb[:, :, i], 98)
        
        rgb[:, :, i] = np.clip(
            (rgb[:, :, i] - p2) / (p98 - p2),
            0, 1
        )
    
    return rgb


def make_false_color(data):
    nir, red, green = 100, 28, 17
    
    fc = np.stack([
        data[:, :, nir],
        data[:, :, red],
        data[:, :, green]
    ], axis=2)
    
    fc = np.where(fc < 0, 0, fc)
    
    for i in range(3):
        p2 = np.nanpercentile(fc[:, :, i], 2)
        p98 = np.nanpercentile(fc[:, :, i], 98)
        
        fc[:, :, i] = np.clip(
            (fc[:, :, i] - p2) / (p98 - p2),
            0, 1
        )
    
    return fc


def save_images(rgb, fc):
    plt.imsave("outputs/rgb.png", rgb)
    plt.imsave("outputs/false_color.png", fc)