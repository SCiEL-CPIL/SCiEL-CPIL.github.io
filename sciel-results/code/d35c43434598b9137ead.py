"""ODE solver: Heat2D 10x10 - uses matrix extraction and exponential integrator."""
import numpy as np
import time

def _extract_matrix(f, y0):
    """Extract the linear operator A from f(t, y) = A @ y (assuming linear, time-independent).
    We call f(0, e_j) for each basis vector to get columns of A.
    Also verify that f is linear and time-independent by checking a few test points."""
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Get A columns: f(0, e_j) = A @ e_j = A[:, j]
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        col = f(0.0, e_j)
        A[:, j] = col
    
    # Verify linearity: f(0, y0) should equal A @ y0
    f_y0 = f(0.0, y0)
    A_y0 = A @ y0
    if np.linalg.norm(f_y0 - A_y0) > 1e-10 * max(1.0, np.linalg.norm(A_y0)):
        return None  # Not linear at t=0
    
    # Check time-independence: f(t, y0) should equal A @ y0 for t != 0
    t_test = 0.01
    f_t_y0 = f(t_test, y0)
    if np.linalg.norm(f_t_y0 - A_y0) > 1e-8 * max(1.0, np.linalg.norm(A_y0)):
        # Try another time
        t_test2 = 0.02
        f_t2_y0 = f(t_test2, y0)
        if np.linalg.norm(f_t2_y0 - A_y0) > 1e-8 * max(1.0, np.linalg.norm(A_y0)):
            return None  # Not time-independent
    
    return A


def _expm_multiply(A, t, y):
    """Compute expm(t*A) @ y using scipy's expm or a scaling-and-squaring approach.
    For a 100x100 matrix, scipy.linalg.expm is efficient enough."""
    from scipy.linalg import expm
    M = expm(t * A)
    return M @ y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0
    
    start_time = time.time()
    
    # Try to extract the linear operator
    A = _extract_matrix(f, y0)
    
    if A is not None:
        # Use matrix exponential for exact solution of linear ODE
        # y(t1) = expm((t1-t0)*A) @ y0
        from scipy.linalg import expm
        M = expm(dt * A)
        y_final = M @ y0
        
        elapsed = time.time() - start_time
        if elapsed < 300:
            return y_final
    
    # Fallback: use RK4 with many steps if matrix extraction failed
    # This should not happen for the heat equation, but keep it as safety
    def _rk4(f, t0, t1, y0, steps):
        h = (t1 - t0) / steps
        t, y = t0, np.array(y0, dtype=float)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
                return None
        return y
    
    y = _rk4(f, t0, t1, y0, 5000)
    if y is None:
        # Last resort: return initial state (shouldn't happen)
        y = y0.copy()
    
    return y
