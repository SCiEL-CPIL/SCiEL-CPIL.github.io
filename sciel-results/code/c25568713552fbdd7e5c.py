"""
ODE solver: 15-dimensional stiff linear system
Strategy: Extract A with 15 calls, then compute expm(dt*A) @ y0 using
a high-precision Taylor series with adaptive scaling to maximize accuracy.
"""
import numpy as np
import time

def _extract_matrix(f, t0, y0):
    """Extract the 15x15 system matrix A using n calls to f."""
    n = len(y0)
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
    return A

def _expm_taylor(M, max_terms=1000, tol=1e-17):
    """
    Compute matrix exponential of M via Taylor series: exp(M) = sum_{k=0}^{inf} M^k/k!
    Uses adaptive stopping based on term magnitude.
    """
    n = M.shape[0]
    result = np.eye(n, dtype=float)
    term = np.eye(n, dtype=float)
    
    # Check if M is zero matrix
    if np.allclose(M, 0.0, atol=1e-15):
        return result
    
    # Compute spectral norm estimate for scaling decision
    # Use Gershgorin-like bound: max row sum of absolute values
    row_sums = np.sum(np.abs(M), axis=1)
    norm_est = np.max(row_sums)
    
    if norm_est < 1e-14:
        return result + M
    
    for k in range(1, max_terms):
        term = term @ M / k
        result = result + term
        
        # Check convergence: norm of term relative to result
        term_norm = np.max(np.sum(np.abs(term), axis=1))
        if term_norm < tol * max(np.max(np.sum(np.abs(result), axis=1)), 1e-300):
            break
    
    return result

def _solve_linear_ode(A, y0, t0, t1):
    """
    Solve y' = A y, y(t0) = y0, at time t1.
    Uses scaling-and-squaring with high-order Taylor series for expm.
    """
    dt = t1 - t0
    if abs(dt) < 1e-15:
        return y0.copy()
    
    M = dt * A
    
    # Check if M is effectively zero
    norm_M = np.max(np.sum(np.abs(M), axis=1))
    if norm_M < 1e-15:
        return y0 + dt * (A @ y0)
    
    # Scaling-and-squaring: choose s such that ||M/2^s||_1 <= 0.5
    # This ensures the Taylor series converges rapidly
    s = 0
    norm_scaled = norm_M
    while norm_scaled > 0.5 and s < 64:
        s += 1
        norm_scaled /= 2.0
    
    # Compute exp(M/2^s) using Taylor series
    M_scaled = M / (2.0 ** s)
    E = _expm_taylor(M_scaled, max_terms=200, tol=1e-18)
    
    # Square s times: E^(2^s) = exp(M)
    for _ in range(s):
        E = E @ E
    
    return E @ y0

def integrate(f, t_span, y0):
    """
    Integrate the 15-dimensional stiff linear ODE y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy:
    1. Extract the system matrix A using exactly n=15 calls to f.
    2. Solve the linear system using scaling-and-squaring with high-order Taylor series.
    
    Total f calls: 15.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Extract A with 15 calls
    A = _extract_matrix(f, t0, y0)
    
    # Check deadline
    if time.time() > deadline:
        return y0
    
    # Solve using scaling-and-squaring with Taylor series
    y_final = _solve_linear_ode(A, y0, t0, t1)
    
    # Sanity check
    if not np.all(np.isfinite(y_final)):
        # Fallback: use scipy expm if available
        try:
            from scipy.linalg import expm
            M = expm((t1 - t0) * A)
            y_final = M @ y0
        except Exception:
            # Last resort: simple RK4
            h = (t1 - t0) / 1000
            y = y0.copy()
            for _ in range(1000):
                k1 = A @ y
                k2 = A @ (y + 0.5 * h * k1)
                k3 = A @ (y + 0.5 * h * k2)
                k4 = A @ (y + h * k3)
                y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            y_final = y
    
    return y_final
