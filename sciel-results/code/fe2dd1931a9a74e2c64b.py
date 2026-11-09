import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multi-scale basis with Chebyshev, reciprocal, and interaction terms.
    
    The 10 parameters are:
    p0: constant
    p1: x
    p2: y
    p3: z
    p4: 1/x
    p5: 1/y
    p6: 1/z
    p7: xy
    p8: xz
    p9: yz
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Reciprocal terms (safe since x, y, z in [0.5, 2])
    inv_x = 1.0 / x
    inv_y = 1.0 / y
    inv_z = 1.0 / z
    
    # Interaction terms
    xy = x * y
    xz = x * z
    yz = y * z
    
    # Build prediction as linear combination
    pred = (p[0] * 1.0 +
            p[1] * x +
            p[2] * y +
            p[3] * z +
            p[4] * inv_x +
            p[5] * inv_y +
            p[6] * inv_z +
            p[7] * xy +
            p[8] * xz +
            p[9] * yz)
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
