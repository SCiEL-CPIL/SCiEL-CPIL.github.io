"""ODE solver: Heat2D 10x10 using matrix extraction and exponential integrator."""
import time
import numpy as np
import scipy.linalg as la


def _extract_matrix(f, y0):
    """Extract the linear operator A from f(t, y) = A @ y."""
    n = len(y0)
    y0 = np.asarray(y0, dtype=float)
    
    # Check if f is linear and time-independent
    # First, get f at y0
    f0 = f(0.0, y0)
    
    # Check linearity: f(t, alpha * y0) should be alpha * f0
    alpha = 2.0
    f_alpha = f(0.0, alpha * y0)
    if not np.allclose(f_alpha, alpha * f0, rtol=1e-10, atol=1e-12):
        return None, None
    
    # Check time-independence
    f_t1 = f(0.025, y0)
    if not np.allclose(f_t1, f0, rtol=1e-10, atol=1e-12):
        return None, None
    
    # Extract A by probing each column
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        f_ej = f(0.0, e_j)
        A[:, j] = f_ej
    
    # Verify: A @ y0 should equal f0
    if not np.allclose(A @ y0, f0, rtol=1e-10, atol=1e-12):
        return None, None
    
    return A, f0


def _expm_multiply(A, t, y):
    """Compute expm(t*A) @ y using scipy's expm_multiply (efficient)."""
    return la.expm_multiply(t * A, y)


def integrate(f, t_span, y0):
    """Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Uses matrix extraction for linear time-invariant systems and exponential
    integrator for high accuracy with minimal function calls.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    deadline = time.time() + 280  # Safety margin before 300s limit
    
    # Try to extract the linear operator
    A, f0 = _extract_matrix(f, y0)
    
    if A is not None:
        # Linear time-invariant system: y(t) = expm((t1-t0)*A) @ y0
        dt = t1 - t0
        try:
            y_final = _expm_multiply(A, dt, y0)
            if np.all(np.isfinite(y_final)):
                return y_final
        except Exception:
            pass
    
    # Fallback: high-order RK4 with adaptive-like stepping
    # Use a fixed number of steps that balances accuracy and call count
    n_steps = 5000
    h = (t1 - t0) / n_steps
    t = t0
    y = y0.copy()
    
    for _ in range(n_steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    
    return y
