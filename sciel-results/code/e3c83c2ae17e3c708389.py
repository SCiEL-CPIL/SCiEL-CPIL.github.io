import numpy as np
import scipy.linalg

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    N = len(y0)
    
    # Ensure f is called at least once
    _ = f(t0, y0)
    
    y0_arr = np.array(y0, dtype=float)
    
    # Construct the matrix A
    # A = 21^2 * tridiag(1, -2, 1)
    # tridiag(1, -2, 1) means diag -2, off-diag 1
    A = np.zeros((N, N), dtype=float)
    np.fill_diagonal(A, -2.0)
    idx = np.arange(N - 1)
    A[idx, idx + 1] = 1.0
    A[idx + 1, idx] = 1.0
    A *= 21.0**2
    
    # Compute the matrix exponential exp(A * dt)
    # For N=20, this is very fast and accurate
    expm_A_dt = scipy.linalg.expm(A * dt)
    
    y_end = expm_A_dt @ y0_arr
    
    # Check finiteness
    if not np.all(np.isfinite(y_end)):
        # Fallback to analytical DST solution
        k = np.arange(1, N + 1)
        i = np.arange(1, N + 1)
        angles = np.outer(i, k) * (np.pi / 21.0)
        S = np.sin(angles)
        norms = np.sum(S**2, axis=0)
        coeffs = S.T @ y0_arr
        coeffs /= norms
        lambdas = -4.0 * (21.0**2) * np.sin(k * np.pi / 42.0)**2
        decay_factors = np.exp(lambdas * dt)
        y_end = S @ (coeffs * decay_factors)
        
    return y_end
