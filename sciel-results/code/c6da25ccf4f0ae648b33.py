import numpy as np

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    n_tubes = N * N
    
    i_idx = np.repeat(np.arange(N), N)
    j_idx = np.tile(np.arange(N), N)
    
    # Star configuration: center shears at c=0.5
    # x_k = -i_k / (2*N)
    # y_k = -j_k / (2*N)
    x_base = -i_idx / (2.0 * N)
    y_base = -j_idx / (2.0 * N)
    
    positions = np.stack([x_base, y_base], axis=1)
    
    return positions
