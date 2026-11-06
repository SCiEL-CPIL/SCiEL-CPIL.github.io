"""Improved program: Regularized degree-5 polynomial fit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.zeros(6)

    # Use Ridge Regression for degree 5 to balance fit quality and generalization
    # This helps avoid overfitting to training noise and improves extrapolation to F=10
    N = 6
    V = np.vander(x, N=N, increasing=True)
    
    # Compute Normal Equations: (V^T V + lambda I) c = V^T y
    A = V.T @ V
    b = V.T @ y
    
    # Choose lambda relative to the scale of A
    # A small lambda allows non-linearity; a larger one stabilizes
    trace_A = np.trace(A)
    if trace_A == 0:
        lambda_val = 1.0
    else:
        lambda_val = 1e-4 * trace_A / N
        
    A_reg = A + lambda_val * np.eye(N)
    
    try:
        coeff = np.linalg.solve(A_reg, b)
    except np.linalg.LinAlgError:
        # Fallback to least squares if solve fails
        coeff, _, _, _ = np.linalg.lstsq(V, y, rcond=None)

    # Safety check
    if time.time() > deadline - safety_margin:
        return coeff

    return coeff
# EVOLVE-BLOCK-END
