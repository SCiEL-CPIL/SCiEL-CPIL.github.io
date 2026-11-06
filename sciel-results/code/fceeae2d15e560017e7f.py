"""
Optimized solver for the 2D Heat Equation (linear system).
Exploits linearity to reconstruct the system matrix A and use matrix exponential
for an exact solution, minimizing function evaluations.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Global deadline guard
_START_TIME = time.time()
_TIMEOUT = 250.0  # Safety margin before 300s/330s limit

def _check_time():
    if time.time() - _START_TIME > _TIMEOUT:
        return True
    return False

def _rk4_fallback(f, t0, t1, y0, steps=10000):
    """High-accuracy RK4 fallback if linearity assumption fails."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        if _check_time():
            return y
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    return y

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # 1. Check for affine term b = f(t, 0)
    y_zero = np.zeros(n)
    f_zero = f(t0, y_zero)
    
    # Heuristic: If f(0) is not close to zero, the system is affine y' = Ay + b.
    # For heat equation, b is typically 0. If b != 0, we handle it or fallback.
    # Let's assume b=0 for the primary path. If it fails consistency, we fallback.
    
    # 2. Reconstruct A assuming y' = A y
    # Evaluate f at basis vectors
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        if _check_time():
            return _rk4_fallback(f, t0, t1, y0)
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
    # 3. Verify linearity and time-invariance
    # Pick a random vector and check if f(t, y) == A y
    # Also check at a different time point to ensure A is constant
    np.random.seed(42)
    y_rand = np.random.randn(n)
    t_mid = 0.5 * (t0 + t1)
    
    # Check at t0
    f_check_0 = f(t0, y_rand)
    A_y_rand = A @ y_rand
    err_0 = np.linalg.norm(f_check_0 - A_y_rand)
    
    # Check at t_mid
    f_check_mid = f(t_mid, y_rand)
    err_mid = np.linalg.norm(f_check_mid - A_y_rand)
    
    # Tolerance for linearity check
    # Scale by magnitude of A and y
    scale = np.linalg.norm(A_y_rand) + 1e-10
    if err_0 < 1e-10 * scale and err_mid < 1e-10 * scale:
        # System is linear and time-invariant: y' = A y
        # Solution: y(t1) = exp(A * (t1 - t0)) * y0
        try:
            from scipy.linalg import expm
            dt = t1 - t0
            exp_A_dt = expm(A * dt)
            y_end = exp_A_dt @ y0
            return y_end
        except Exception:
            # Fallback if expm fails
            return _rk4_fallback(f, t0, t1, y0)
    else:
        # System is not purely linear homogeneous, or time-dependent.
        # Fallback to high-accuracy explicit RK4.
        # 10000 steps is very accurate for this small problem.
        return _rk4_fallback(f, t0, t1, y0, steps=10000)
# EVOLVE-BLOCK-END
