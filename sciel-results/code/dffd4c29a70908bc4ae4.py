"""Symbolic regression with 2 inputs using a hybrid polynomial + Fourier basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 110.0  # 10s safety margin before 120s limit

    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Construct a hybrid basis with 10 terms.
    # We combine orthogonal polynomial terms (Legendre) for global trend
    # and Fourier terms for oscillatory behavior.
    #
    # Strategy:
    # - Use Legendre polynomials P0, P1, P2 for x and y (orthogonal on [-1,1])
    # - Include key interaction terms
    # - Include a few Fourier terms for potential oscillations
    #
    # Basis selection (10 terms):
    # 0: 1 (constant)
    # 1: x (linear in x)
    # 2: y (linear in y)
    # 3: P2(x) = 0.5*(3x^2 - 1) (quadratic in x, orthogonal)
    # 4: P2(y) = 0.5*(3y^2 - 1) (quadratic in y, orthogonal)
    # 5: xy (interaction)
    # 6: x*P2(y) (mixed quadratic)
    # 7: P2(x)*y (mixed quadratic)
    # 8: sin(x + y) (oscillatory interaction)
    # 9: cos(x - y) (oscillatory interaction)
    
    # Compute powers
    x2 = x * x
    y2 = y * y
    
    # Legendre Polynomials
    # P0 = 1
    # P1 = x, y
    # P2 = 0.5 * (3z^2 - 1)
    p2_x = 0.5 * (3.0 * x2 - 1.0)
    p2_y = 0.5 * (3.0 * y2 - 1.0)
    
    # Term 0: Constant
    term_0 = np.ones_like(x)
    
    # Term 1: x
    term_1 = x
    
    # Term 2: y
    term_2 = y
    
    # Term 3: P2(x)
    term_3 = p2_x
    
    # Term 4: P2(y)
    term_4 = p2_y
    
    # Term 5: xy
    term_5 = x * y
    
    # Term 6: x * P2(y)
    term_6 = x * p2_y
    
    # Term 7: P2(x) * y
    term_7 = p2_x * y
    
    # Term 8: sin(x + y)
    term_8 = np.sin(x + y)
    
    # Term 9: cos(x - y)
    term_9 = np.cos(x - y)
    
    # Stack terms into a matrix of shape (n_points, 10)
    basis = np.column_stack([
        term_0, term_1, term_2, term_3, term_4,
        term_5, term_6, term_7, term_8, term_9
    ])
    
    # The model is linear in params: pred = basis @ params
    # BFGS will optimize params to minimize MSE.
    # Since it's linear, this is effectively a least squares problem,
    # but BFGS is robust enough to find it quickly from any start.
    
    predictions = basis @ params
    
    # Ensure output is finite and correct shape
    if not np.all(np.isfinite(predictions)):
         # Fallback to zero if something went wrong (should not happen)
         return np.zeros_like(x)
        
    return predictions


# EVOLVE-BLOCK-END
