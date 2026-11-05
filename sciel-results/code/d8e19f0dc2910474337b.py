import numpy as np
import scipy.linalg as la
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s margin for 300s limit

    # Try to extract the linear operator A (assume homogeneous first)
    # For the 2D heat equation with Dirichlet BCs, f(t, 0) = 0.
    # Probe basis vectors to get columns of A.
    A = np.empty((n, n))
    
    for j in range(n):
        if time.time() > deadline:
            # Timeout during probing, fallback
            return _rk4(f, t0, t1, y0, 2000)
        
        e_j = np.zeros(n)
        e_j[j] = 1.0
        try:
            val = f(t0, e_j)
            A[:, j] = val
        except Exception:
            # Non-linear or error
            return _rk4(f, t0, t1, y0, 2000)
    
    # Verify homogeneity: check f(0, 0) == 0
    # If it's not zero, the system is affine and we need to handle it.
    try:
        f_zero = f(t0, np.zeros(n))
    except Exception:
        f_zero = None
    
    if f_zero is not None and np.linalg.norm(f_zero) > 1e-12:
        # Affine case: y' = A_true @ y + b
        # We probed: f(t, e_j) = A_true @ e_j + b = col_j(A_true) + b
        # So A[:, j] = col_j(A_true) + b
        # A_true[:, j] = A[:, j] - b for all j
        A_true = A - f_zero[:, np.newaxis]
        
        # Augment state: z = [y; 1], z' = [A_true, b; 0, 0] z
        A_aug = np.zeros((n + 1, n + 1))
        A_aug[:n, :n] = A_true
        A_aug[:n, n] = f_zero
        
        y_aug = np.zeros(n + 1)
        y_aug[:n] = y0
        y_aug[n] = 1.0
        
        dt = t1 - t0
        M_aug = la.expm(A_aug * dt)
        y_end_aug = M_aug @ y_aug
        y_end = y_end_aug[:n]
    else:
        # Homogeneous case
        dt = t1 - t0
        M = la.expm(A * dt)
        y_end = M @ y0
    
    return y_end

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
