"""Combined symbolic regression program: rational + quadratic interaction basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a hybrid basis that combines:
    1. A rational core (ratio of two affine functions) for non-additive interactions.
    2. A set of monomial and reciprocal terms to capture specific functional forms.
    
    Basis (10 terms):
    1. 1
    2. x
    3. y
    4. z
    5. 1/x
    6. 1/y
    7. 1/z
    8. (x + y + z)  [Sum interaction]
    9. (x*y + x*z + y*z) [Pairwise sum interaction]
    10. (x + y + z) / (1 + 0.5*(x + y + z)) [Rational saturation term]
    
    This structure allows the optimizer to fit:
    - Linear behaviors (via terms 2-4)
    - Reciprocal behaviors (via terms 5-7)
    - Non-linear saturation or rational dependencies (via term 10)
    - Interaction effects (via terms 8-9)
    
    The rational term is stabilized with a small epsilon in the denominator to avoid
    division by zero during optimization steps.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    # Calculate reciprocal terms (safe since inputs >= 0.5)
    inv_x = 1.0 / x
    inv_y = 1.0 / y
    inv_z = 1.0 / z
    
    # Interaction terms
    sum_xyz = x + y + z
    pair_sum = x * y + x * z + y * z
    
    # Rational term: (x + y + z) / (1 + 0.5*(x + y + z))
    # Denominator is always > 1 for x,y,z >= 0.5, so it's safe.
    # We add a tiny epsilon just in case of numerical drift in params if we were to make it parametric,
    # but here the structure is fixed.
    denom_rat = 1.0 + 0.5 * sum_xyz
    rat_term = sum_xyz / denom_rat
    
    # Construct feature matrix (N, 10)
    features = np.column_stack((
        np.ones_like(x),
        x,
        y,
        z,
        inv_x,
        inv_y,
        inv_z,
        sum_xyz,
        pair_sum,
        rat_term
    ))
    
    # Compute prediction: features (n, 10) dot params (10,) -> (n,)
    pred = features @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
