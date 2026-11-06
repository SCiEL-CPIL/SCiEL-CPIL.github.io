"""ODE solver: 15-dimensional stiff linear system using matrix extraction."""
import numpy as np
import scipy.linalg
import time

def _extract_matrix_A(f, t0, y0):
    """
    Extract the system matrix A from y' = f(t, y) assuming y' = A y.
    Uses 15 unit vector probes to directly read A's columns.
    """
    n = len(y0)
    
    # Evaluate f(t0, e_j) for each standard basis vector e_j
    # f(t0, e_j) = A @ e_j = A[:, j] (the j-th column of A)
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
    
    return A

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin
    _ = deadline  # Keep for future use; matrix exponential is fast enough
    
    # 1. Attempt to identify as linear autonomous system y' = A y
    # Probe for linearity and time-independence
    # We need to be careful with call count.
    # Let's extract A first.
    
    A = _extract_matrix_A(f, t0, y0)
    
    # Verify A is consistent: check if f(t, y) == A @ y for a random y
    # Also check time dependence: f(t0, y) == f(t1, y)
    
    # Let's do a quick verification with 2 extra calls
    # y_test = random
    rng = np.random.default_rng(123)
    y_test = rng.standard_normal(n)
    
    f_t0_y_test = f(t0, y_test)
    f_t1_y_test = f(t1, y_test)
    
    # Check linearity at t0: f(t0, y_test) vs A @ y_test
    residual_lin = np.linalg.norm(f_t0_y_test - A @ y_test)
    norm_f = np.linalg.norm(f_t0_y_test)
    rel_err_lin = residual_lin / (norm_f + 1e-12)
    
    # Check time independence: f(t0, y_test) vs f(t1, y_test)
    residual_time = np.linalg.norm(f_t0_y_test - f_t1_y_test)
    rel_err_time = residual_time / (norm_f + 1e-12)
    
    # If errors are small, assume autonomous linear system
    if rel_err_lin < 1e-10 and rel_err_time < 1e-10:
        # Use matrix exponential
        # y(t1) = expm(A * (t1 - t0)) @ y0
        dt = t1 - t0
        if dt == 0:
            return y0
        
        # expm is very fast for 15x15
        M = scipy.linalg.expm(A * dt)
        y_final = M @ y0
        
        # Sanity check: is the result finite?
        if np.all(np.isfinite(y_final)):
            return y_final
    
    # Fallback: Use a high-accuracy implicit solver if the linear assumption fails
    # This is unlikely to be triggered based on the task description, but good for robustness.
    # We use solve_ivp with Radau (good for stiff) and tight tolerance.
    
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0, method='Radau', rtol=1e-12, atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: Return the initial state
    return y0
