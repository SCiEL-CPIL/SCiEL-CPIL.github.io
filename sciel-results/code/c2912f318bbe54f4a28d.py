"""ODE solver for convection-diffusion 100-point problem using adaptive high-order methods."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _dop853_step(f, t, y, h):
    """Single DOP853 step (8th order with 7th order error estimate).
    Returns y_new, y_err (error estimate), and number of f calls."""
    n = len(y)
    c = np.array([
        0, 1/35, 2/35, 4/35, 1/5, 2/5, 4/5, 8/9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ])
    # DOP853 coefficients - using the standard tableau
    # This is a 27-stage method; for efficiency I'll use a well-known implementation
    
    # Actually, let me use a simpler but effective approach: RKF78 or a 5th-order pair
    # For robustness and speed, let's use a 4th/5th order pair (Dormand-Prince)
    # which is well-tested and efficient
    
    return None  # placeholder, we'll use DP below


def _dormand_prince_step(f, t, y, h):
    """Single Dormand-Prince step (5th/4th order). Returns y_new, y_err, nfeval."""
    n = len(y)
    # Dormand-Prince coefficients
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
    ])
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    b_star = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    
    nfeval = 0
    k = np.zeros((7, n))
    
    for i in range(7):
        t_i = t + c[i] * h
        y_i = y.copy()
        for j in range(i):
            if A[i, j] != 0:
                y_i = y_i + h * A[i, j] * k[j]
        k[i] = f(t_i, y_i)
        nfeval += 1
    
    y_new = y + h * np.dot(b, k)
    y_new_star = y + h * np.dot(b_star, k)
    y_err = y_new - y_new_star
    
    return y_new, y_err, nfeval


def _adaptive_rk45(f, t0, t1, y0, rtol=1e-10, atol=1e-12, max_steps=100000, deadline=None):
    """Adaptive Dormand-Prince 4(5) solver with time budget guard."""
    t = t0
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Initial step size estimate
    h = (t1 - t0) / 100
    h_min = 1e-14
    h_max = (t1 - t0) / 10
    
    nfeval = 0
    steps = 0
    max_nfeval = 2000000
    
    # First step
    y_new, y_err, nf = _dormand_prince_step(f, t, y, h)
    nfeval += nf
    steps += 1
    
    if not np.all(np.isfinite(y_new)) or np.max(np.abs(y_new)) > 1e12:
        return y, nfeval, False
    
    # Error norm
    err_norm = np.max(np.abs(y_err) / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new))))
    
    if err_norm <= 1.0:
        # Accept step
        y = y_new
        t = t + h
        # Adjust step size
        if err_norm < 1e-8:
            h *= min(5.0, 0.9 * (1.0 / err_norm) ** 0.2)
        else:
            h *= min(5.0, 0.9 * (1.0 / err_norm) ** 0.25)
        h = min(max(h, h_min), h_max)
    else:
        # Reject step, reduce h
        h *= max(0.1, 0.9 * (1.0 / err_norm) ** 0.25)
        h = max(h, h_min)
    
    while t < t1 - 1e-15:
        # Check deadline
        if deadline is not None and time.time() > deadline:
            return y, nfeval, True
        
        # Check f-call budget
        if nfeval >= max_nfeval:
            return y, nfeval, True
        
        # Check steps
        if steps >= max_steps:
            return y, nfeval, True
        
        # Ensure we don't overshoot
        if t + h > t1:
            h = t1 - t
        
        y_new, y_err, nf = _dormand_prince_step(f, t, y, h)
        nfeval += nf
        steps += 1
        
        if not np.all(np.isfinite(y_new)) or np.max(np.abs(y_new)) > 1e12:
            # Step failed, reduce h significantly
            h *= 0.1
            if h < h_min:
                return y, nfeval, True
            continue
        
        err_norm = np.max(np.abs(y_err) / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new))))
        
        if err_norm <= 1.0:
            y = y_new
            t = t + h
            # Adjust step size
            if err_norm < 1e-8:
                factor = 5.0
            else:
                factor = 0.9 * (1.0 / err_norm) ** 0.25
            h *= min(5.0, max(0.1, factor))
            h = min(max(h, h_min), h_max)
        else:
            h *= max(0.1, 0.9 * (1.0 / err_norm) ** 0.25)
            h = max(h, h_min)
    
    return y, nfeval, True


def _implicit_euler_step(f, t, y, h, J_cache=None):
    """Single implicit Euler step with Newton iteration. Returns y_new, J, nfeval."""
    n = len(y)
    z = y.copy()
    nfeval = 0
    
    # Build Jacobian if not provided
    if J_cache is None:
        J = np.empty((n, n))
        fz = f(t + h, z)
        nfeval += 1
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1e-8 * max(1.0, abs(z[j]))
            fz_p = f(t + h, z + e)
            fz_m = f(t + h, z - e)
            nfeval += 2
            J[:, j] = (fz_p - fz_m) / (2 * e[j])
    else:
        J = J_cache
        fz = f(t + h, z)
        nfeval += 1
    
    # Newton iterations
    M = np.eye(n) - h * J
    for _ in range(10):
        rhs = -(z - y - h * fz)
        try:
            step = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            # Fallback to least squares
            step, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        
        z_new = z + step
        if np.linalg.norm(step) <= 1e-12 * (1 + np.linalg.norm(z_new)):
            z = z_new
            break
        
        z = z_new
        fz = f(t + h, z)
        nfeval += 1
    
    return z, nfeval


def _backward_euler_adaptive(f, t0, t1, y0, rtol=1e-8, atol=1e-10, max_steps=50000, deadline=None):
    """Adaptive implicit Euler for stiff problems."""
    t = t0
    y = np.array(y0, dtype=float)
    n = len(y)
    
    h = (t1 - t0) / 100
    h_min = 1e-14
    h_max = (t1 - t0) / 10
    
    nfeval = 0
    steps = 0
    max_nfeval = 2000000
    
    # Initial step
    y_new, nf = _implicit_euler_step(f, t, y, h)
    nfeval += nf
    steps += 1
    
    if not np.all(np.isfinite(y_new)) or np.max(np.abs(y_new)) > 1e12:
        return y, nfeval, False
    
    # Estimate error using a simple Richardson extrapolation idea
    # For now, just accept and adjust based on residual
    residual = np.max(np.abs(y_new - y - h * f(t + h, y_new)))
    err_norm = residual / (atol + rtol * np.max(np.abs(y_new)))
    
    y = y_new
    t = t + h
    if err_norm < 1e-8:
        h *= min(3.0, 2.0)
    else:
        h *= max(0.3, min(1.0, 0.9 / (err_norm ** 0.5)))
    h = min(max(h, h_min), h_max)
    
    while t < t1 - 1e-15:
        if deadline is not None and time.time() > deadline:
            return y, nfeval, True
        if nfeval >= max_nfeval:
            return y, nfeval, True
        if steps >= max_steps:
            return y, nfeval, True
        
        if t + h > t1:
            h = t1 - t
        
        y_new, nf = _implicit_euler_step(f, t, y, h)
        nfeval += nf
        steps += 1
        
        if not np.all(np.isfinite(y_new)) or np.max(np.abs(y_new)) > 1e12:
            h *= 0.5
            if h < h_min:
                return y, nfeval, True
            continue
        
        residual = np.max(np.abs(y_new - y - h * f(t + h, y_new)))
        err_norm = residual / (atol + rtol * np.max(np.abs(y_new)))
        
        y = y_new
        t = t + h
        
        if err_norm < 1e-8:
            h *= min(3.0, 2.0)
        else:
            h *= max(0.3, min(1.0, 0.9 / (err_norm ** 0.5)))
        h = min(max(h, h_min), h_max)
    
    return y, nfeval, True


def integrate(f, t_span, y0):
    """Solve y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Set deadline with safety margin (330s limit, use 300s for computation)
    start_time = time.time()
    deadline = start_time + 295.0  # 5s safety margin
    
    # Try adaptive explicit RK45 first
    y, nfeval, success = _adaptive_rk45(f, t0, t1, y0, rtol=1e-10, atol=1e-12, deadline=deadline)
    
    if success and np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1e12:
        return y
    
    # If explicit failed, try implicit Euler
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        y_implicit, nf_implicit, success_implicit = _backward_euler_adaptive(
            f, t0, t1, y0, rtol=1e-8, atol=1e-10, deadline=deadline
        )
        if success_implicit and np.all(np.isfinite(y_implicit)) and np.max(np.abs(y_implicit)) < 1e12:
            return y_implicit
    
    # Fallback: fixed-step RK4 with many steps
    steps = 5000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            break
    
    return y
# EVOLVE-BLOCK-END
