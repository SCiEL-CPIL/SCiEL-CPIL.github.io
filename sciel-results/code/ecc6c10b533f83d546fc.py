import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Combine Chebyshev polynomials (for smooth polynomial-like behavior)
    # with trigonometric functions (for oscillatory behavior)
    # This gives a richer basis than either alone, while remaining well-conditioned
    
    # Scale x to [-1, 1] for Chebyshev stability
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_4 (5 terms)
    T = np.zeros((x.shape[0], 5))
    T[:, 0] = 1.0
    if x.shape[0] > 0:
        T[:, 1] = t
        for k in range(2, 5):
            T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
    
    # Compute trigonometric basis functions (5 terms)
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    x_sin_x = x * sin_x
    
    # Build the 10-column basis matrix
    # Columns 0-4: Chebyshev T_0..T_4
    # Columns 5-9: sin(x), cos(x), sin(2x), cos(2x), x*sin(x)
    basis = np.hstack([
        T,
        sin_x[:, np.newaxis],
        cos_x[:, np.newaxis],
        sin_2x[:, np.newaxis],
        cos_2x[:, np.newaxis],
        x_sin_x[:, np.newaxis]
    ])
    
    # Linear combination
    pred = basis @ params
    
    # Safety guard: replace non-finite values
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
