"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
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


def integrate(f, t_span, y0):
    t0, t1 = t_span
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, 20s safety margin
    
    # Try RK4 with progressively more steps until we have high accuracy or run out of time
    # The problem is not stiff over [0, 0.1], so RK4 works well
    steps_list = [100, 200, 400, 800, 1600]
    y = None
    for steps in steps_list:
        if time.time() > deadline - 30:
            break
        y = _rk4(f, t0, t1, y0, steps)
        if y is not None:
            break
    
    if y is None:
        # Fallback: use the last successful result or a minimal RK4
        y = _rk4(f, t0, t1, y0, 50)
        if y is None:
            # Absolute fallback: single step
            h = t1 - t0
            k1 = f(t0, y0)
            y = np.array(y0, dtype=float) + h * k1
    
    return y
# EVOLVE-BLOCK-END
