"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _dop54(f, t0, t1, y0, steps):
    """DOP54 (Dormand-Prince 5(4,3)) explicit Runge-Kutta method."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    
    # DOP54 coefficients
    c = [0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0]
    a = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0],
         [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0],
         [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0],
         [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0]]
    b5 = [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0]
    
    for _ in range(steps):
        k = [None] * 7
        for i in range(7):
            y_stage = y.copy()
            for j in range(i):
                y_stage = y_stage + h * a[i][j] * k[j]
            k[i] = f(t + c[i] * h, y_stage)
        
        y_new = y.copy()
        for i in range(7):
            y_new = y_new + h * b5[i] * k[i]
        y = y_new
        t += h
        
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = _dop54(f, t0, t1, y0, 100)
    if y is None:
        y = _dop54(f, t0, t1, y0, 200)
    return y
# EVOLVE-BLOCK-END
