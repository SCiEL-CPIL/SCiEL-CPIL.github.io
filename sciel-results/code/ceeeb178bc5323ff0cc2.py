"""Rewritten program for symbolic regression with mixed polynomial/rational basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a mixed basis of polynomial and rational terms.
    
    The basis consists of 10 terms:
    1, x, y, z, x*y, x*z, y*z, 1/x, 1/y, 1/z
    
    This basis allows the model to capture linear, interaction, and inverse relationships.
    Inputs are in [0.5, 2], so 1/x is safe and well-conditioned.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
        
    # Construct the feature matrix (n_samples, 10)
    # Using column_stack for efficiency and clarity
    features = np.column_stack((
        np.ones_like(x),
        x,
        y,
        z,
        x * y,
        x * z,
        y * z,
        1.0 / x,
        1.0 / y,
        1.0 / z
    ))
    
    # Compute prediction: features (n, 10) dot params (10,) -> (n,)
    pred = features @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
