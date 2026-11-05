"""ODE solver: Convection-diffusion 100-point using matrix exponential for linear systems."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time


def _detect_linear_time_invariant(f, t0, t1, y0):
    """
    Detect if f(t, y) = A @ y + b (linear time-invariant).
    Returns (A, b) if detected, else None.
    """
    n = len(y0)
    
    # Test time-invariance first: f(t0, y) should equal f(t1, y)
    y_test = np.array(y0, dtype=float).copy()
    f_t0 = f(t0, y_test)
    f_t1 = f(t1, y_test)
    if not np.allclose(f_t0, f_t1, rtol=1e-10, atol=1e-12):
        return None
    
    # Test linearity: f(t, y1 + y2) should equal f(t, y1) + f(t, y2)
    # Use a random vector
    rng = np.random.RandomState(42)
    v = rng.randn(n)
    f_y0 = f(t0, y_test)
    f_v = f(t0, v)
    f_y0_plus_v = f(t0, y_test + v)
    
    # Check if f is linear (through origin) or affine
    # If linear: f(y0 + v) = f(y0) + f(v)
    # If affine: f(y0 + v) = f(y0) + f(v) - f(0)
    
    # First try pure linear: f(0) should be 0
    f_zero = f(t0, np.zeros(n))
    
    if np.allclose(f_zero, np.zeros(n), rtol=1e-10, atol=1e-12):
        # Pure linear case
        if not np.allclose(f_y0_plus_v, f_y0 + f_v, rtol=1e-8, atol=1e-10):
            return None
        # f(t, y) = A @ y
        # Build A by probing with basis vectors
        A = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A[:, j] = f(t0, e_j)
        b = np.zeros(n)
    else:
        # Affine case: f(t, y) = A @ y + b
        # f(y0 + v) = A(y0 + v) + b = A*y0 + A*v + b = f(y0) - b + A*v + b = f(y0) + A*v
        # So f(y0 + v) - f(y0) = A * v
        # And f(v) - b = A * v, so f(y0 + v) - f(y0) = f(v) - b
        # Check: f(y0 + v) - f(y0) should equal f(v) - b
        # But we don't know b yet. Let's use: f(y0 + v) - f(y0) = A*v
        # And f(v) = A*v + b, so A*v = f(v) - b
        # Thus f(y0 + v) - f(y0) = f(v) - b
        # => b = f(v) - (f(y0 + v) - f(y0))
        
        # Verify consistency with another random vector
        v2 = rng.randn(n)
        f_v2 = f(t0, v2)
        f_y0_plus_v2 = f(t0, y_test + v2)
        
        b_from_v = f_v - (f_y0_plus_v - f_y0)
        b_from_v2 = f_v2 - (f_y0_plus_v2 - f_y0)
        
        if not np.allclose(b_from_v, b_from_v2, rtol=1e-8, atol=1e-10):
            return None
        
        b = b_from_v
        # Now A*v = f(v) - b
        A = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A[:, j] = f(t0, e_j) - b
    
    return A, b


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280  # Safety margin from 300s limit
    
    # Try to detect if the system is linear time-invariant
    result = _detect_linear_time_invariant(f, t0, t1, y0)
    
    if result is not None:
        A, b = result
        T = t1 - t0
        
        # Solve y' = A*y + b using matrix exponential
        # y(t) = e^{At} y0 + ∫_0^t e^{A(t-s)} b ds
        # For constant b: ∫_0^t e^{A(t-s)} b ds = ∫_0^t e^{Au} b du (substituting u = t-s)
        # = (e^{At} - I) A^{-1} b if A is invertible
        # More robustly: use the formula y(t) = e^{At} y0 + ∫_0^T e^{A(T-s)} b ds
        # We can compute this as: expand the integral using block matrix exponential
        # [y(T); c] = expm([[A, b], [0, 0]]) @ [y0; T]
        
        # Build augmented system for exact solution
        # Let z = [y; 1] where the last component is 1 (constant)
        # Then z' = [A y + b; 0] = [[A, b], [0, 0]] @ z
        # with z(0) = [y0; 1]
        
        n_aug = n + 1
        M = np.zeros((n_aug, n_aug))
        M[:n, :n] = A
        M[:n, n] = b
        # Last row is all zeros
        
        z0 = np.zeros(n_aug)
        z0[:n] = y0
        z0[n] = 1.0
        
        # Compute matrix exponential
        M_exp = expm(M * T)
        z_T = M_exp @ z0
        
        y_final = z_T[:n]
        
        if not np.all(np.isfinite(y_final)):
            # Fallback to RK4
            return _rk4_fallback(f, t0, t1, y0, 2000, deadline)
        
        return y_final
    
    else:
        # Fallback to RK4
        return _rk4_fallback(f, t0, t1, y0, 2000, deadline)


def _rk4_fallback(f, t0, t1, y0, steps, deadline=None):
    """RK4 with deadline guard."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    
    for i in range(steps):
        if deadline is not None and time.time() > deadline:
            return y
        
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            break
    
    return y
# EVOLVE-BLOCK-END
