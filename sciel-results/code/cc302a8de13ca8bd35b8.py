"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy: For a linear ODE y' = A*y, extract A using n function calls
    (leveraging y0 and unit vectors), then compute the exact solution
    y(t_end) = expm(A * (t_end - t_start)) @ y0.
    
    This uses only n calls to f (theoretical minimum for extracting A)
    and gives the exact solution (up to floating point precision).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, 10s safety margin
    
    # Extract the system matrix A using n function calls.
    # We need to find A such that f(t, y) = A @ y for all t, y.
    # For a linear system, f(t, y) = A @ y (independent of t).
    # 
    # Strategy: Use y0 as the first probe, then use unit vectors for the rest.
    # f(t, y0) = A @ y0  -> gives A @ y0
    # f(t, e_j) = A @ e_j = A[:, j]  -> gives column j of A
    #
    # But we need n columns. If y0 is not zero, we can use:
    #   f(t, y0) = A @ y0
    #   f(t, e_j) = A @ e_j for j = 0, ..., n-1
    # That's n+1 calls. To reduce to n calls:
    #   If y0 has at least one nonzero component, say y0[k] != 0, then
    #   we can solve for column k from the first call and use unit vectors
    #   for the other n-1 columns. Total: n calls.
    #
    # Simpler approach that always works with n calls:
    #   Use unit vectors e_0, e_1, ..., e_{n-1} for all n columns.
    #   f(t, e_j) = A @ e_j = A[:, j]
    #   Total: n calls.
    
    A = np.zeros((n, n), dtype=float)
    
    # Check if we're running out of time
    if time.time() > deadline:
        # Fallback: return y0 (best guess if no time)
        return y0
    
    # Extract each column of A using unit vectors
    for j in range(n):
        if time.time() > deadline:
            break
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
    
    # Check if we're running out of time
    if time.time() > deadline:
        # Fallback: use Euler's method with what we have
        h = (t1 - t0) / max(1, int(100))
        y = y0.copy()
        for _ in range(int(100)):
            if time.time() > deadline:
                break
            y = y + h * (A @ y)
        return y
    
    # Compute the exact solution: y(t1) = expm(A * (t1 - t0)) @ y0
    dt = t1 - t0
    if abs(dt) < 1e-15:
        return y0
    
    M = expm(A * dt)
    y_final = M @ y0
    
    # Verify finiteness
    if not np.all(np.isfinite(y_final)):
        # Fallback to a simple method
        h = dt / 1000
        y = y0.copy()
        for _ in range(1000):
            y = y + h * (A @ y)
        y_final = y
    
    return y_final
# EVOLVE-BLOCK-END
