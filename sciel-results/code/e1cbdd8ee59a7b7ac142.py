# EVOLVE-BLOCK-START
import numpy as np
import time


def _estimate_jacobian(f, t, y, n):
    """Estimate the Jacobian of f at (t, y) using forward differences."""
    J = np.empty((n, n), dtype=float)
    fz = f(t, y)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        eps = 1e-7 * max(1.0, abs(y[j]))
        e[j] = eps
        J[:, j] = (f(t, y + e) - fz) / eps
    return J


def _implicit_step(f, t, y, h, n, J_prev=None, max_newton=10, tol=1e-12):
    """
    One step of implicit Euler (backward Euler) with Newton iteration.
    Solves: y_{n+1} = y_n + h * f(t + h, y_{n+1})
    Uses Jacobian from previous step if available (secant-like reuse).
    """
    t_new = t + h
    z = y.copy()
    
    # Initial Jacobian estimate
    if J_prev is not None:
        J = J_prev.copy()
    else:
        J = _estimate_jacobian(f, t_new, z, n)
    
    for _ in range(max_newton):
        fz = f(t_new, z)
        # Solve: (I - h*J) dz = -(z - y - h*fz)
        rhs = -(z - y - h * fz)
        try:
            step = np.linalg.solve(np.eye(n) - h * J, rhs)
        except np.linalg.LinAlgError:
            # Fallback to least squares or just break
            break
        z = z + step
        if np.linalg.norm(step) <= tol * (1 + np.linalg.norm(z)):
            break
    
    # Update Jacobian for next step (optional, but helps convergence)
    J_new = _estimate_jacobian(f, t_new, z, n)
    return z, J_new


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    total_time = t1 - t0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 300.0  # 300s limit
    
    # Adaptive step size control
    # Start with a reasonable step size
    h = total_time / 100.0
    min_h = total_time / 10000.0  # Max 10000 steps
    max_h = total_time / 10.0     # Min 10 steps
    
    t = t0
    J_prev = None
    steps_taken = 0
    max_steps = 100000  # Safety cap on total steps
    
    while t < t1 - 1e-15 and steps_taken < max_steps:
        # Check deadline
        if time.time() > deadline - 2.0:
            break
        
        # Ensure we don't overshoot t1
        h_eff = min(h, t1 - t)
        
        # Try implicit Euler step
        y_new, J_prev = _implicit_step(f, t, y, h_eff, n, J_prev)
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(y_new)):
            # Reduce step size and retry
            h *= 0.5
            if h < min_h:
                # If step is too small, just accept the bad result or break
                break
            continue
        
        # Adaptive step size based on error estimate
        # Use a simple heuristic: if the change is large, reduce h
        change = np.linalg.norm(y_new - y)
        scale = max(1.0, np.linalg.norm(y))
        rel_change = change / scale
        
        if rel_change > 1e-2:
            # Too large a change, reduce step
            h *= 0.5
            h = max(h, min_h)
            # Don't advance t, retry with smaller h
            continue
        elif rel_change < 1e-6:
            # Small change, can increase step
            h *= 1.5
            h = min(h, max_h)
        
        # Accept step
        y = y_new
        t += h_eff
        steps_taken += 1
    
    return y
# EVOLVE-BLOCK-END
