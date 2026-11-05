import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280

    # 1. Check time-invariance
    # Sample a few points
    t_mid = (t0 + t1) / 2
    y_test = np.random.RandomState(42).randn(n)
    y_test = y_test / np.linalg.norm(y_test) * 0.1 # Small amplitude to avoid non-linearities if any
    
    f_t0 = f(t0, y_test)
    f_t1 = f(t1, y_test)
    f_tmid = f(t_mid, y_test)
    
    # Check if f(t, y) is independent of t
    # Use a relative tolerance
    if not (np.allclose(f_t0, f_t1, rtol=1e-8, atol=1e-10) and 
            np.allclose(f_t0, f_tmid, rtol=1e-8, atol=1e-10)):
        # Time-dependent, use high-precision adaptive solver
        return _high_precision_solve(f, t_span, y0)

    # 2. Check linearity
    # f(t, y) = A @ y + b
    b = f(t0, np.zeros(n))
    
    # Extract A
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        f_ej = f(t0, e_j)
        A[:, j] = f_ej - b
        
    # Verify linearity with a random vector
    y_rand = np.random.RandomState(123).randn(n)
    f_rand = f(t0, y_rand)
    f_pred = A @ y_rand + b
    
    if not np.allclose(f_rand, f_pred, rtol=1e-8, atol=1e-10):
        # Non-linear, use high-precision adaptive solver
        return _high_precision_solve(f, t_span, y0)

    # 3. Solve linear time-invariant system using Matrix Exponential
    # y(t1) = expm(A*dt) @ y0 + integral_0^dt expm(A*s) @ b ds
    # Use augmented matrix
    Z = np.zeros((n + 1, n + 1))
    Z[:n, :n] = A
    Z[:n, n] = b
    
    try:
        M_aug = expm(Z * dt)
        y_end = M_aug[:n, :n] @ y0 + M_aug[:n, n]
        
        if not np.all(np.isfinite(y_end)):
            return _high_precision_solve(f, t_span, y0)
            
        return y_end
        
    except Exception:
        return _high_precision_solve(f, t_span, y0)

def _high_precision_solve(f, t_span, y0):
    """
    High precision adaptive solver for general or non-linear/time-dependent cases.
    """
    # DOP853 is 8th order, very accurate
    # Use very tight tolerances to maximize score
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-13, 
        atol=1e-15,
        max_step=(t_span[1] - t_span[0]) / 50 # Ensure enough resolution
    )
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback to RK4 with many steps if solve_ivp fails
        return _rk4(f, t_span[0], t_span[1], y0, 10000)

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
            return y
    return y
