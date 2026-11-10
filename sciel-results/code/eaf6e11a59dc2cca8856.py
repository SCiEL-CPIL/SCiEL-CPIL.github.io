"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1]
    t = x / 3.0
    
    # Construct mixed Fourier/Constant basis
    # 10 parameters:
    # 0: Constant (1)
    # 1: sin(t)
    # 2: cos(t)
    # 3: sin(2t)
    # 4: cos(2t)
    # 5: sin(3t)
    # 6: cos(3t)
    # 7: sin(4t)
    # 8: cos(4t)
    # 9: sin(5t)
    
    basis = np.empty((x.shape[0], 10))
    basis[:, 0] = 1.0
    
    for k in range(1, 6):
        idx_sin = 2 * k - 1
        idx_cos = 2 * k
        
        if idx_sin < 10:
            basis[:, idx_sin] = np.sin(k * t)
        if idx_cos < 10:
            basis[:, idx_cos] = np.cos(k * t)
            
    # Note: The loop above sets:
    # k=1: idx_sin=1, idx_cos=2
    # k=2: idx_sin=3, idx_cos=4
    # k=3: idx_sin=5, idx_cos=6
    # k=4: idx_sin=7, idx_cos=8
    # k=5: idx_sin=9, idx_cos=10 (out of bounds, so only sin(5t) is set at 9)
    
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
