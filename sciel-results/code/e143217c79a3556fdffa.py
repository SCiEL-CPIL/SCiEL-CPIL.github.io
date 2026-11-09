import numpy as np

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    i_idx = np.repeat(np.arange(N), N)
    j_idx = np.tile(np.arange(N), N)
    
    # Diagonal compensation: fully counteract shear
    # x_k = -i/N so that x_k + c*i/N + 1/N is constant in c at right edge
    x_base = -i_idx / N
    y_base = -j_idx / N
    
    positions = np.stack([x_base, y_base], axis=1)
    
    return positions
