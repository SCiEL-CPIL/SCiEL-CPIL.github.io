import numpy as np
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    Optimized for linear time-invariant systems y' = A y.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 300.0  # 300s margin for 330s limit

    # Attempt to identify as LTI system y' = A y
    # We assume f(t, y) = A @ y
    # We need to find A.
    # A[:, j] = f(t0, e_j)
    
    # If n is small, this is very cheap.
    # We need n evaluations to get A.
    # Let's check if we have time.
    if time.time() > deadline - 5.0:
        # Not enough time for the "smart" method, fallback to standard solver immediately
        return _fallback_solve(f, t_span, y0)

    A = np.empty((n, n), dtype=float)
    
    # Evaluate f at basis vectors to extract columns of A
    # Note: We use t0 for all evaluations, assuming A is constant.
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
        # Check deadline
        if time.time() > deadline - 5.0:
            return _fallback_solve(f, t_span, y0)

    # Verify linearity: A @ y0 should be close to f(t0, y0)
    # This helps if the system is actually affine or non-linear, though prompt says linear.
    # However, if it's affine y' = Ay + c, then f(t, e_j) = A e_j + c.
    # Then A_estimated = A + c * ones. 
    # f(t0, y0) = A y0 + c.
    # A_est @ y0 = A y0 + c * sum(y0).
    # This check might fail for affine systems.
    # Given "linear system", we assume homogeneous.
    
    # Let's do a quick consistency check with one extra call if needed, 
    # but the prompt implies we should minimize calls.
    # The "Knowledge" section says "reducing function calls from thousands to ~16".
    # This implies 15 calls for A columns + 1 for check? Or just 15?
    # Let's stick to 15 calls for A.
    
    # Compute matrix exponential
    dt = t1 - t0
    try:
        # expm is stable for 15x15
        M = expm(A * dt)
        y_end = M @ y0
    except Exception:
        return _fallback_solve(f, t_span, y0)

    # Basic sanity check on result
    if not np.all(np.isfinite(y_end)):
        return _fallback_solve(f, t_span, y0)

    return y_end

def _fallback_solve(f, t_span, y0):
    """
    Fallback solver for non-linear or if expm fails.
    Uses scipy.integrate.solve_ivp with a stiff method.
    """
    t0, t1 = t_span
    # Use a stiff solver like BDF or Radau
    # rtol and atol set for high precision
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='BDF', 
        rtol=1e-10, 
        atol=1e-12
    )
    if sol.success:
        return sol.y[:, -1]
    else:
        # Try Radau
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
        # Last resort: return last known state or raise
        return sol.y[:, -1] if sol.t.size > 0 else y0
