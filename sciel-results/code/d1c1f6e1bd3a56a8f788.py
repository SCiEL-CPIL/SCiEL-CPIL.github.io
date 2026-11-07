"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Bounded tanh-based model with 10 parameters.
    Uses a combination of linear terms, interaction, and bounded non-linearities.
    
    Basis:
    0: constant
    1: x
    2: y
    3: x*y
    4: tanh(x)
    5: tanh(y)
    6: tanh(x+y)
    7: tanh(x-y)
    8: tanh(x*y)
    9: tanh(x^2 + y^2)
    """
    p = params
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    t1 = np.tanh(x)
    t2 = np.tanh(y)
    t3 = np.tanh(x + y)
    t4 = np.tanh(x - y)
    t5 = np.tanh(x * y)
    t6 = np.tanh(x * x + y * y)
    
    pred = (
        p[0] +
        p[1] * x +
        p[2] * y +
        p[3] * (x * y) +
        p[4] * t1 +
        p[5] * t2 +
        p[6] * t3 +
        p[7] * t4 +
        p[8] * t5 +
        p[9] * t6
    )
    
    return pred
# EVOLVE-BLOCK-END
