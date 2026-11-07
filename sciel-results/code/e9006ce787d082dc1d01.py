"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid symbolic regression model using a rich linear basis.
    Combines polynomials, reciprocals, exponentials, logs, trigonometric,
    and Gaussian RBF components. The model is linear in the 10 parameters,
    allowing BFGS to converge efficiently to the global optimum for this basis.
    
    Basis functions (10 total):
    0: Constant (1)
    1: Linear (x)
    2: Quadratic (x^2)
    3: Cubic (x^3)
    4: Quartic (x^4)
    5: Reciprocal (1/x)
    6: Exponential (exp(x)) - clipped for stability
    7: Logarithmic (log(x))
    8: Gaussian RBF at x=2.0, sigma=1.0
    9: Gaussian RBF at x=4.0, sigma=1.0
    
    Prediction: y = sum_{i=0}^{9} params[i] * basis_i(x)
    """
    deadline = time.time() + 100  # Safety margin
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure params are finite
    if not np.all(np.isfinite(p)):
        p = np.where(np.isfinite(p), p, 1.0)
        
    n = x.shape[0]
    
    # Safe x for log and reciprocal
    x_safe = np.clip(x, 1e-10, None)
    
    # Compute basis terms
    # 0: Constant
    b0 = 1.0
    
    # 1: Linear
    b1 = x
    
    # 2: Quadratic
    b2 = x * x
    
    # 3: Cubic
    b3 = b2 * x
    
    # 4: Quartic
    b4 = b2 * b2
    
    # 5: Reciprocal
    b5 = 1.0 / x_safe
    
    # 6: Exponential (clipped to avoid overflow)
    b6 = np.exp(np.clip(x, -10, 10))
    
    # 7: Logarithmic
    b7 = np.log(x_safe)
    
    # 8: Gaussian RBF at center 2.0, sigma 1.0
    # exp(-0.5 * (x - 2.0)^2 / 1.0^2)
    diff1 = x - 2.0
    b8 = np.exp(-0.5 * diff1 * diff1)
    
    # 9: Gaussian RBF at center 4.0, sigma 1.0
    # exp(-0.5 * (x - 4.0)^2 / 1.0^2)
    diff2 = x - 4.0
    b9 = np.exp(-0.5 * diff2 * diff2)
    
    # Linear combination
    prediction = (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        p[8] * b8 +
        p[9] * b9
    )
    
    # Check for non-finite values
    if not np.all(np.isfinite(prediction)):
        prediction = np.where(np.isfinite(prediction), prediction, 0.0)
        
    return prediction
# EVOLVE-BLOCK-END
