"""Combined ODE solver for 10x10 2D heat equation."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as la
import time


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


def _crank_nicolson(f, t0, t1, y0, steps, J):
    """
    Crank-Nicolson using a precomputed Jacobian J.
    Assumes f(t, y) = J @ y (time-independent linear system).
    For the heat equation, this is exact if J is the Laplacian.
    """
    h = (t1 - t0) / steps
    n = len(y0)
    
    # Precompute matrices
    I = np.eye(n)
    A = I - 0.5 * h * J
    B = I + 0.5 * h * J
    
    # LU factorization of A
    lu, piv = la.lu_factor(A)
    
    y = np.array(y0, dtype=float)
    for _ in range(steps):
        rhs = B.dot(y)
        y = la.lu_solve((lu, piv), rhs)
        if not np.all(np.isfinite(y)):
            return None
    return y


def _estimate_jacobian(f, t0, y0):
    """Estimate the Jacobian of f with respect to y at (t0, y0)."""
    n = len(y0)
    fy0 = f(t0, y0)
    J = np.zeros((n, n))
    
    for j in range(n):
        e = np.zeros(n)
        # Use a scale based on the value magnitude
        scale = 1e-7 * max(1.0, abs(y0[j]))
        e[j] = scale
        J[:, j] = (f(t0, y0 + e) - fy0) / scale
    
    return J


def integrate(f, t_span, y0):
    deadline = time.time() + 290.0  # Safety margin for 300s limit
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)
    
    # Try Crank-Nicolson with estimated Jacobian first.
    # This is much more efficient for linear stiff problems like heat eq.
    # We need to be careful: if f is not exactly linear, or if J is wrong, this fails.
    # But for the heat equation, f(t,y) = L*y, so J=L is constant.
    
    # Estimate J
    # This costs n+1 evaluations of f.
    J = _estimate_jacobian(f, t0, y0_arr)
    
    # Check if the system looks linear and time-independent by verifying f(t0, y0) == J @ y0
    # If not, fall back to RK4.
    fy0_check = f(t0, y0_arr)
    if np.linalg.norm(fy0_check - J.dot(y0_arr)) > 1e-5 * np.linalg.norm(fy0_check):
        # Non-linear or time-dependent, use RK4
        y = _rk4(f, t0, t1, y0_arr, 5000)
        if y is None:
            # Fallback to a simple explicit method with small steps if RK4 fails
            # But for heat eq, RK4 should be stable for reasonable steps.
            # Let's try a larger number of steps for RK4 if it failed? No, just return best effort.
            return y0_arr
        return y
    
    # Use Crank-Nicolson
    # How many steps?
    # CN is 2nd order. Error ~ h^2.
    # Let's try 100 steps first. If we have time, we can do more?
    # No, CN is unconditionally stable. More steps = more accuracy but more cost.
    # The score is -log10(err) - 0.5*log10(calls).
    # Each CN step uses 0 calls to f (since we use J).
    # The initial J estimation costs n+1 = 101 calls.
    # So total calls = 101.
    # If we use RK4 with 5000 steps, calls = 20000.
    # log10(101) ~ 2, log10(20000) ~ 4.3.
    # Difference in penalty: 0.5 * (4.3 - 2) = 1.15.
    # So CN is favored by ~1.15 in the score if accuracy is similar.
    
    # Let's use 200 steps for CN to ensure high accuracy.
    steps = 200
    y = _crank_nicolson(f, t0, t1, y0_arr, steps, J)
    
    if y is not None:
        return y
    
    # Fallback if CN failed (shouldn't happen for linear system)
    y = _rk4(f, t0, t1, y0_arr, 5000)
    if y is None:
        return y0_arr
    return y
# EVOLVE-BLOCK-END
