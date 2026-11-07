"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a function using a bivariate Fourier-like basis.
    This uses exactly 10 parameters corresponding to a selection of
    trigonometric basis functions:
    1, cos(x), sin(x), cos(y), sin(y),
    cos(x)cos(y), sin(x)sin(y), cos(x+y), sin(x+y), cos(2x)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Ensure p has length 10
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    else:
        p = p[:10]
        
    # Define the 10 basis functions
    # Order: 
    # 0: 1
    # 1: cos(x)
    # 2: sin(x)
    # 3: cos(y)
    # 4: sin(y)
    # 5: cos(x)cos(y)
    # 6: sin(x)sin(y)
    # 7: cos(x+y)
    # 8: sin(x+y)
    # 9: cos(2x)
    
    pred = np.zeros_like(x)
    pred += p[0] * 1.0
    pred += p[1] * np.cos(x)
    pred += p[2] * np.sin(x)
    pred += p[3] * np.cos(y)
    pred += p[4] * np.sin(y)
    pred += p[5] * np.cos(x) * np.cos(y)
    pred += p[6] * np.sin(x) * np.sin(y)
    pred += p[7] * np.cos(x + y)
    pred += p[8] * np.sin(x + y)
    pred += p[9] * np.cos(2 * x)
    
    return pred
# EVOLVE-BLOCK-END
