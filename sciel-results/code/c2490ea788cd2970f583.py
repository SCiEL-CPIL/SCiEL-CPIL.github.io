"""
Structural change: Product-of-sums decomposition with internal nonlinear fitting.

The model is f(x, y) = (sum_{i=0}^{4} a_i * phi_i(x)) * (sum_{j=0}^{4} b_j * psi_j(y))
where phi_i and psi_j are basis functions (Chebyshev-like polynomials).

The 10 params are interpreted as:
  params[0:5] -> initial values for a (x-coefficients)
  params[5:10] -> initial values for b (y-coefficients)

Internally, we perform a nonlinear least-squares fit (via scipy.optimize.least_squares)
to find the best (a, b) that minimize the MSE on the training data.
This decouples the structural search (basis choice) from the coefficient optimization.
We use a deadline guard to ensure we finish within the time limit.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

# Deadline guard
_start_time = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _chebyshev_basis(x, n_terms):
    """
    Compute Chebyshev polynomials of the first kind T_k(x) for k=0..n_terms-1.
    x is expected to be in [-1, 1].
    Returns array of shape (len(x), n_terms).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    basis = np.zeros((n, n_terms))
    if n_terms > 0:
        basis[:, 0] = 1.0
    if n_terms > 1:
        basis[:, 1] = x
    for k in range(2, n_terms):
        basis[:, k] = 2.0 * x * basis[:, k-1] - basis[:, k-2]
    return basis

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _start_time
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback: return linear model if we're out of time
        # This is just a safety net; shouldn't happen in normal operation
        return params[0] * x + params[1] * y + params[2]
    
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    params = np.asarray(params, dtype=float).ravel()
    
    n_points = len(x)
    if n_points == 0:
        return np.array([])
    
    # Basis size: 5 terms each for x and y
    n_basis = 5
    
    # Precompute basis matrices
    X_basis = _chebyshev_basis(x, n_basis)  # shape (n_points, 5)
    Y_basis = _chebyshev_basis(y, n_basis)  # shape (n_points, 5)
    
    # Initial guess for coefficients from params
    a0 = params[0:5].copy()
    b0 = params[5:10].copy()
    
    # Ensure initial guess is not all zeros
    if np.allclose(a0, 0):
        a0 = np.ones(5)
    if np.allclose(b0, 0):
        b0 = np.ones(5)
    
    def residuals(coeffs):
        a = coeffs[:5]
        b = coeffs[5:]
        # f(x,y) = (X_basis @ a) * (Y_basis @ b)
        fx = X_basis @ a  # shape (n_points,)
        fy = Y_basis @ b  # shape (n_points,)
        pred = fx * fy
        return pred - y
    
    # Initial coefficients
    coeffs0 = np.concatenate([a0, b0])
    
    try:
        # Use least_squares for nonlinear fit
        result = least_squares(
            residuals, 
            coeffs0, 
            method='lm',  # Levenberg-Marquardt, good for this size
            max_nfev=1000
        )
        
        best_coeffs = result.x
        a_best = best_coeffs[:5]
        b_best = best_coeffs[5:]
        
        # Compute predictions
        fx = X_basis @ a_best
        fy = Y_basis @ b_best
        predictions = fx * fy
        
        # Check for finite values
        if not np.all(np.isfinite(predictions)):
            # Fallback to initial guess predictions
            fx = X_basis @ a0
            fy = Y_basis @ b0
            predictions = fx * fy
            if not np.all(np.isfinite(predictions)):
                predictions = np.zeros(n_points)
                
        return predictions
        
    except Exception:
        # Fallback: return predictions from initial guess
        try:
            fx = X_basis @ a0
            fy = Y_basis @ b0
            predictions = fx * fy
            if not np.all(np.isfinite(predictions)):
                predictions = np.zeros(n_points)
            return predictions
        except Exception:
            return np.zeros(n_points)

# EVOLVE-BLOCK-END
