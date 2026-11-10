import numpy as np

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    n_tubes = N * N
    
    i_idx = np.repeat(np.arange(N), N)
    j_idx = np.tile(np.arange(N), N)
    
    # Corner configuration: base at origin
    # x_k = 0
    # y_k = 0
    x_base = np.zeros(n_tubes)
    y_base = np.zeros(n_tubes)
    
    positions = np.stack([x_base, y_base], axis=1)
    
    return positions
