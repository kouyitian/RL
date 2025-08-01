import numpy as np
from scipy.ndimage import gaussian_filter1d

def gaussian_smooth(points, sigma=2.0):
    pts = np.array(points)
    x = gaussian_filter1d(pts[:,0], sigma=sigma)
    y = gaussian_filter1d(pts[:,1], sigma=sigma)
    return list(zip(x.tolist(), y.tolist()))
