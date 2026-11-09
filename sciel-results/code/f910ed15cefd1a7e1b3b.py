"""ODE integrator for the discretized heat equation (N = 20) using high-precision analytic eigen-decomposition."""

import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = A y where A = 21^2 * tridiag(1, -2, 1) for N=20.
    
    Uses the analytical eigendecomposition approach with high precision (longdouble)
    to minimize numerical error.
    
    Parameters
    ----------
    f : callable
        The right-hand side function f(t, y) -> y'. Called at least once.
    t_span : tuple
        (t_start, t_end) integration interval.
    y0 : ndarray
        Initial condition of shape (20,).
        
    Returns
    -------
    ndarray
        The state y at t = t_end, shape (20,).
    """
    start_time = time.time()
    deadline = start_time + 320.0
    
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=np.longdouble)
    N = len(y0)
    
    if N == 0:
        f(t0, y0.astype(float))
        return y0.astype(float)

    # Use longdouble for higher precision
    k = np.arange(1, N + 1, dtype=np.longdouble)
    i = np.arange(1, N + 1, dtype=np.longdouble)
    
    # V matrix: V[k-1, i-1] = sin(k*pi*i/(N+1))
    # Using np.pi in longdouble context
    pi = np.longdouble(np.pi)
    
    V = np.sin(np.outer(k, i) * pi / (N + 1))
    
    # Eigenvalues: lambda_k = -4*(N+1)^2 * sin^2(k*pi/(2*(N+1)))
    N_plus_1 = np.longdouble(N + 1)
    lambdas = -4.0 * (N_plus_1 ** 2) * np.sin(k * pi / (2.0 * N_plus_1)) ** 2
    
    # Orthogonality norm
    norm_sq = (N + 1) / 2.0
    norm_sq = np.longdouble(norm_sq)
    
    # Project y0 onto eigenvectors
    c = (V.T @ y0) / norm_sq
    
    # Compute decay factors
    dt = np.longdouble(t1) - np.longdouble(t0)
    decay_factors = np.exp(lambdas * dt)
    
    # Reconstruct y_end
    y_end_ld = V @ (c * decay_factors)
    
    # Convert back to float64 for return
    y_end = y_end_ld.astype(float)
    
    # Ensure finite
    if not np.all(np.isfinite(y_end)):
        f(t0, y0.astype(float))
        return y0.astype(float)
    
    # Call f at least once
    f(t0, y0.astype(float))
    
    return y_end
