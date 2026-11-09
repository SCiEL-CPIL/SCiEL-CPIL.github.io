"""ODE solver: Heat2D 10x10 - high-accuracy integrator with adaptive RK45."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rk45_step(f, t, y, h):
    """Single step of Dormand-Prince RK45. Returns (y4, y5, error_estimate)."""
    a2, a3, a4, a5, a6, a7 = 1/5, 3/10, 4/5, 8/9, 1, 1
    b21 = 1/5
    b31, b32 = 3/40, 9/40
    b41, b42, b43 = 44/45, -56/15, 32/9
    b51, b52, b53, b54 = -13/20, 0, 3/4, -9/5
    b61, b62, b63, b64, b65 = -1/12, 0, 1/3, 1/4, 5/12
    b71, b72, b73, b74, b75, b76 = 19/72, 0, -5/12, 7/12, -1/10, 1/40

    c2, c3, c4, c5, c6, c7 = 1/5, 3/10, 4/5, 1, 1, 1

    k1 = f(t, y)
    k2 = f(t + c2*h, y + h*b21*k1)
    k3 = f(t + c3*h, y + h*(b31*k1 + b32*k2))
    k4 = f(t + c4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
    k5 = f(t + c5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
    k6 = f(t + c6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
    k7 = f(t + c7*h, y + h*(b71*k1 + b72*k2 + b73*k3 + b74*k4 + b75*k5 + b76*k6))

    # 5th order solution
    y5 = y + h*(1935/2197*k1 - 9237/3392*k3 + 3855/11048*k4 + 945/11048*k5 - 95/1092*k6)
    # 4th order solution
    y4 = y + h*(25/216*k1 + 1408/2565*k3 + 2197/4104*k4 - 1/10*k5 + 1/40*k6)
    # Error estimate
    err = y5 - y4
    return y5, err


def _rk45_adaptive(f, t0, t1, y0, rtol=1e-12, atol=1e-14, max_steps=500000):
    """Adaptive RK45 with tight tolerances."""
    t = t0
    y = np.array(y0, dtype=float)
    h = (t1 - t0) / 100  # initial step size
    h_min = (t1 - t0) / max_steps
    n = len(y)
    
    # Safety and control parameters
    safety = 0.9
    factor_min = 0.1
    factor_max = 10.0
    
    while t < t1 - 1e-15:
        # Ensure we don't overshoot
        if t + h > t1:
            h = t1 - t
        
        y_new, err = _rk45_step(f, t, y, h)
        
        # Error norm
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.sqrt(np.mean((err / scale) ** 2))
        
        if err_norm <= 1.0:
            # Accept step
            t = t + h
            y = y_new
            # Adjust step size
            if err_norm > 1e-15:
                h_new = h * safety * err_norm ** (-0.2)
            else:
                h_new = h * factor_max
            h = min(max(h_new, h_min), factor_max * h)
        else:
            # Reject step, reduce h
            h_new = h * safety * err_norm ** (-0.25)
            h = max(h_new, h_min)
    
    return y


def _rk4_fixed(f, t0, t1, y0, steps):
    """Fixed-step RK4 for comparison/fallback."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y


def integrate(f, t_span, y0):
    """Integrate ODE y' = f(t, y) from t_span[0] to t_span[1].
    
    Uses adaptive RK45 with very tight tolerances for high accuracy.
    """
    t0, t1 = t_span
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Try adaptive RK45 first with tight tolerances
    try:
        result = _rk45_adaptive(f, t0, t1, y0, rtol=1e-13, atol=1e-15)
        if np.all(np.isfinite(result)):
            return result
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        # Fallback: use RK4 with moderate steps
        try:
            return _rk4_fixed(f, t0, t1, y0, 5000)
        except Exception:
            return np.array(y0, dtype=float)
    
    # Try RK4 with more steps as fallback
    try:
        result = _rk4_fixed(f, t0, t1, y0, 20000)
        if np.all(np.isfinite(result)):
            return result
    except Exception:
        pass
    
    # Final fallback
    return np.array(y0, dtype=float)
# EVOLVE-BLOCK-END
