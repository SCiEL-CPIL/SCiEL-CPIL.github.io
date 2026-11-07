"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Precompute Chebyshev basis functions for x and y
# T_0(t) = 1
# T_1(t) = t
# T_2(t) = 2t^2 - 1
# T_3(t) = 4t^3 - 3t
# T_4(t) = 8t^4 - 8t^2 + 1

def _chebyshev_basis(t):
    """Return Chebyshev polynomial values up to degree 4 for input t."""
    t = np.asarray(t, dtype=np.float64)
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    return np.column_stack([T0, T1, T2, T3, T4])

def _build_design_matrix(x, y):
    """
    Build a design matrix using tensor products of Chebyshev polynomials.
    We use degrees up to 2 for each variable to keep the parameter count manageable
    while capturing smooth interactions.
    
    Basis terms:
    T_i(x) * T_j(y) for i in [0,1,2], j in [0,1,2]
    This gives 9 terms. We have 10 params, so we can add one more, e.g., T_3(x)*T_0(y) or similar.
    Let's use a slightly richer set to utilize all 10 params.
    
    Let's use:
    i in [0,1,2], j in [0,1,2] -> 9 terms
    Plus T_2(x)*T_2(y) is already included.
    Let's add T_3(x)*T_1(y) to make 10 terms? Or just use a fixed set of 10.
    
    Actually, let's stick to a clean structure. 
    Degrees 0,1,2 for x (3 terms) and 0,1,2 for y (3 terms) gives 9 products.
    We have 10 params. Let's add T_3(x)*T_0(y) = T_3(x).
    
    Terms:
    0: T0(x)*T0(y) = 1
    1: T1(x)*T0(y) = x
    2: T2(x)*T0(y) = 2x^2-1
    3: T0(x)*T1(y) = y
    4: T1(x)*T1(y) = xy
    5: T2(x)*T1(y) = (2x^2-1)y
    6: T0(x)*T2(y) = 2y^2-1
    7: T1(x)*T2(y) = x(2y^2-1)
    8: T2(x)*T2(y) = (2x^2-1)(2y^2-1)
    9: T3(x)*T0(y) = 4x^3-3x
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Compute Chebyshev polynomials for x
    Tx0 = np.ones_like(x)
    Tx1 = x
    Tx2 = 2.0*x**2 - 1.0
    Tx3 = 4.0*x**3 - 3.0*x
    
    # Compute Chebyshev polynomials for y
    Ty0 = np.ones_like(y)
    Ty1 = y
    Ty2 = 2.0*y**2 - 1.0
    
    # Build the 10 feature columns
    F0 = Tx0 * Ty0
    F1 = Tx1 * Ty0
    F2 = Tx2 * Ty0
    F3 = Tx0 * Ty1
    F4 = Tx1 * Ty1
    F5 = Tx2 * Ty1
    F6 = Tx0 * Ty2
    F7 = Tx1 * Ty2
    F8 = Tx2 * Ty2
    F9 = Tx3 * Ty0
    
    return np.column_stack([F0, F1, F2, F3, F4, F5, F6, F7, F8, F9])

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y using a linear combination of Chebyshev polynomial features.
    The params are fitted by the evaluator using BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure params has length 10
    if len(params) < 10:
        # Pad with zeros if necessary, though evaluator should provide 10
        params = np.pad(params, (0, 10 - len(params)))
    else:
        params = params[:10]
    
    # Build design matrix
    X = _build_design_matrix(x, y)
    
    # Compute prediction: X @ params
    prediction = X @ params
    
    # Ensure prediction is finite
    if not np.all(np.isfinite(prediction)):
        # Return zeros or a safe fallback if non-finite
        return np.zeros_like(x)
    
    return prediction
# EVOLVE-BLOCK-END
