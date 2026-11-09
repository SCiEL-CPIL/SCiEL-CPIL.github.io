import numpy as np

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    i_idx = np.repeat(np.arange(N), N)
    j_idx = np.tile(np.arange(N), N)
    
    x_base = -i_idx / (2.0 * N)
    y_base = -j_idx / (2.0 * N)
    
    return np.stack([x_base, y_base], axis=1)
