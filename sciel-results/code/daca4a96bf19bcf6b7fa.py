import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    """
    Integrate the 15-dimensional stiff linear ODE y' = A y from t_span[0] to t_span[1].
    
    Strategy:
    1. Extract the system matrix A using exactly n=15 calls to f.
       Since the system is linear and autonomous (implied by "fixed 15-dimensional stiff linear system"),
       f(t, y) = A @ y. We probe with the standard basis vectors e_1, ..., e_n.
       f(t0, e_j) = A @ e_j = column j of A.
       Total calls: 15.
       
    2. Solve the linear system analytically using the matrix exponential:
       y(t1) = exp((t1 - t0) * A) @ y0
       
    This approach minimizes the number of function evaluations (maximizing the -0.5*log10(calls) term)
    and achieves near-machine-precision accuracy (maximizing the -log10(error) term).
    
    Score = -log10(relative error) - 0.5 * log10(evaluations)
    With 15 evaluations and ~1e-15 relative error:
    Score ≈ 15 - 0.5 * log10(15) ≈ 15 - 0.5 * 1.176 ≈ 14.41
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # If the time span is zero, return y0 immediately
    if t1 == t0:
        return y0
    
    # Extract matrix A using n calls with standard basis vectors
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
        # Check deadline
        if time.time() > deadline:
            # If we're running out of time, try to compute with what we have
            # But since we need all columns, just return a best guess
            # In practice, this shouldn't happen for 15 calls
            pass
    
    # Compute the matrix exponential solution
    dt = t1 - t0
    try:
        M = expm(dt * A)
        y_final = M @ y0
        if np.all(np.isfinite(y_final)):
            return y_final
    except Exception:
        pass
    
    # Fallback: if expm fails, try a simple Euler method with many steps
    # (This should rarely be needed)
    try:
        y = y0.copy()
        n_steps = 10000
        h = dt / n_steps
        for _ in range(n_steps):
            y = y + h * f(t0, y)
            if time.time() > deadline:
                break
        return y
    except Exception:
        pass
    
    # Last resort
    return y0
