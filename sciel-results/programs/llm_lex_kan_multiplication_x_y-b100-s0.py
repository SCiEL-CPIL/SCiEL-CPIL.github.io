"""Symbolic regression with 2 inputs using a rich, well-conditioned basis."""
import numpy as np
from scipy.optimize import minimize


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict using a 2D Chebyshev polynomial basis up to total degree 3,
    plus radial and angular terms to capture broader function structure.
    
    The basis is designed to be well-conditioned so BFGS converges reliably.
    We use Chebyshev polynomials for the polynomial part (better conditioned
    than monomials) and add radial/angular terms for non-polynomial structure.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    
    # Chebyshev polynomials
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Radial and angular components
    r2 = x**2 + y**2
    r = np.sqrt(r2)
    theta = np.arctan2(y, x)
    
    # 10 basis functions combining Chebyshev and radial-angular terms
    b0 = T0_x * T0_y          # 1
    b1 = T1_x * T0_y          # x
    b2 = T0_x * T1_y          # y
    b3 = T2_x * T0_y          # 2x^2 - 1
    b4 = T0_x * T2_y          # 2y^2 - 1
    b5 = T1_x * T1_y          # xy
    b6 = T3_x * T0_y          # 4x^3 - 3x
    b7 = T0_x * T3_y          # 4y^3 - 3y
    b8 = r2 * np.cos(2 * theta)   # radial-angular
    b9 = r * np.sin(3 * theta)    # higher-order radial-angular
    
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Clip basis values to avoid extreme values that could cause numerical issues
    basis = np.clip(basis, -10, 10)
    
    # Prediction is linear combination
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
