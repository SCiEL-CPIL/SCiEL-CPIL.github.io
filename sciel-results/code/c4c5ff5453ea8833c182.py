import numpy as np
import scipy.linalg

def _extract_matrix(f, n):
    """Extract the system matrix A from f assuming f(t, y) = A y."""
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        # Use t=0 as the system is autonomous
        A[:, j] = f(0.0, e_j)
    return A

def _check_linearity(f, A, y0, n):
    """Check if f(t, y) is approximately A y."""
    # Test with a few random vectors
    np.random.seed(42)
    for _ in range(3):
        v = np.random.randn(n)
        y_test = f(0.0, v)
        a_test = A @ v
        if not np.allclose(y_test, a_test, rtol=1e-10, atol=1e-12):
            return False
    return True

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
    return y

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Attempt linear system solution
    try:
        A = _extract_matrix(f, n)
        if _check_linearity(f, A, y0, n):
            # Compute matrix exponential
            M = scipy.linalg.expm(A * dt)
            y_end = M @ y0
            # Sanity check for finite values
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Fallback to RK4 if linear assumption fails
    # Use more steps for better accuracy in fallback
    return _rk4(f, t0, t1, y0, 5000)
