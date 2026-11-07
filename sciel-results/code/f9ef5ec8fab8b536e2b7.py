"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    # 6 polynomial terms for global structure
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # 4 RBF terms at corners for local flexibility
    sigma = 0.7
    r1 = np.sqrt((x - 1.0)**2 + (y - 1.0)**2)
    r2 = np.sqrt((x + 1.0)**2 + (y + 1.0)**2)
    r3 = np.sqrt((x - 1.0)**2 + (y + 1.0)**2)
    r4 = np.sqrt((x + 1.0)**2 + (y - 1.0)**2)
    
    g1 = np.exp(-r1**2 / (2 * sigma**2))
    g2 = np.exp(-r2**2 / (2 * sigma**2))
    g3 = np.exp(-r3**2 / (2 * sigma**2))
    g4 = np.exp(-r4**2 / (2 * sigma**2))
    
    pred = (
        p[0] +
        p[1] * x +
        p[2] * y +
        p[3] * x2 +
        p[4] * y2 +
        p[5] * xy +
        p[6] * g1 +
        p[7] * g2 +
        p[8] * g3 +
        p[9] * g4
    )
    return pred
# EVOLVE-BLOCK-END
