"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a function using a bivariate polynomial of degree 3.
    This uses exactly 10 parameters corresponding to the 10 monomials
    of degree <= 3 in two variables:
    1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
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
    # 1: x
    # 2: y
    # 3: x^2
    # 4: xy
    # 5: y^2
    # 6: x^3
    # 7: x^2 y
    # 8: x y^2
    # 9: y^3
    
    pred = np.zeros_like(x)
    pred += p[0] * 1.0
    pred += p[1] * x
    pred += p[2] * y
    pred += p[3] * x**2
    pred += p[4] * x * y
    pred += p[5] * y**2
    pred += p[6] * x**3
    pred += p[7] * x**2 * y
    pred += p[8] * x * y**2
    pred += p[9] * y**3
    
    return pred
# EVOLVE-BLOCK-END
