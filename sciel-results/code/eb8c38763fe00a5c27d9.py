"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    n = x.size
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Build a rich set of basis functions using all 10 params
    # We'll create a feature matrix and solve for optimal params via least_squares
    
    # Scale inputs to roughly [-1, 1] for better conditioning
    x_s = (x - 1.25) / 0.75  # center 1.25, range 1.5 -> [-1, 1]
    y_s = (y - 1.25) / 0.75
    z_s = (z - 1.25) / 0.75
    
    # Basis functions: constant, linear, quadratic, some cross terms
    # 10 basis functions to match 10 params
    basis = np.column_stack([
        np.ones_like(x_s),
        x_s,
        y_s,
        z_s,
        x_s * y_s,
        x_s * z_s,
        y_s * z_s,
        x_s ** 2,
        y_s ** 2,
        z_s ** 2
    ])
    
    # Try to find better params using the 'y' argument as target if it looks like a target
    # Wait - the signature is equation(x, y, z, params). 
    # In the seed: return params[0]*x + params[1]*y + params[2]*z + params[3]
    # This treats x, y, z as inputs. So y is an input, not the target.
    # The evaluator fits params by minimizing training MSE, but equation only returns predictions.
    # So we just need to return a good prediction using the basis and params.
    
    # Since we can't see the target, we just compute the linear combination
    pred = basis @ params
    
    # Clip to avoid extreme values
    pred = np.clip(pred, -1e6, 1e6)
    
    return pred
# EVOLVE-BLOCK-END
