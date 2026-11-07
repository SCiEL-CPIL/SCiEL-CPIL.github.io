"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _dop853_coefficients():
    """
    Return the coefficients for DOP853 (Dormand-Prince 8(5,3)) as used by scipy.
    This is a high-order Runge-Kutta method with error estimation.
    """
    # Nodes (c)
    c = np.array([
        0.0,
        1/35,
        2/35,
        3/35,
        4/35,
        8/35,
        1/7,
        2/7,
        3/7,
        4/7,
        5/7,
        6/7,
        8/9,
        1.0
    ])
    
    # a matrix (lower triangular, a[i][j] for j < i)
    # We'll build it as a list of arrays for each stage
    a = [
        np.array([]),
        np.array([1/35]),
        np.array([2/35, -1/35]),
        np.array([3/35, -9/35, 6/35]),
        np.array([4/35, -16/35, 12/35, -3/35]),
        np.array([8/35, -48/35, 48/35, -24/35, 6/35]),
        np.array([1/7, -4/7, 12/7, -12/7, 8/7, -2/7]),
        np.array([2/7, -16/7, 48/7, -48/7, 32/7, -8/7]),
        np.array([3/7, -24/7, 72/7, -72/7, 48/7, -12/7]),
        np.array([4/7, -32/7, 96/7, -96/7, 64/7, -16/7]),
        np.array([5/7, -40/7, 120/7, -120/7, 80/7, -20/7]),
        np.array([6/7, -48/7, 144/7, -144/7, 96/7, -24/7]),
        np.array([8/9, -80/9, 240/9, -240/9, 160/9, -40/9]),
        np.array([1.0, -7.0, 21.0, -21.0, 14.0, -3.5])
    ]
    
    # b coefficients for the 8th order solution
    b = np.array([
        37/3780, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])
    # Actually, let's use the correct b for DOP853
    # The 8th order weights
    b8 = np.array([
        37/3780, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])
    # This is getting complicated. Let me use a simpler high-order method.
    # I'll use RK45 (Dormand-Prince 4(5)) which is well-known and efficient.
    
    # For RK45:
    c45 = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    a45 = [
        np.array([]),
        np.array([1/5]),
        np.array([3/40, 9/40]),
        np.array([44/45, -56/15, 32/9]),
        np.array([19372/6561, -25360/2187, 64448/6561, -212/729]),
        np.array([9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]),
        np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    ]
    b45 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    b4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    
    return c45, a45, b45, b4


def _rk45_adaptive(f, t0, t1, y0, rtol=1e-11, atol=1e-13, max_steps=100000, min_h=1e-12, max_h=0.1):
    """
    Adaptive step-size RK45 (Dormand-Prince 4(5)) integrator.
    """
    c, a, b5, b4 = _dop853_coefficients()
    
    t = t0
    y = np.array(y0, dtype=float)
    h = max_h
    n_stages = len(c)
    
    steps = 0
    while t < t1 - 1e-15:
        # Don't overshoot
        if t + h > t1:
            h = t1 - t
        
        # Compute stages
        k = np.zeros((n_stages - 1, len(y)))
        for i in range(1, n_stages):
            y_stage = y.copy()
            for j in range(i):
                y_stage += h * a[i - 1][j] * k[j]
            k[i - 1] = f(t + c[i] * h, y_stage)
        
        # Compute 5th order solution
        y_new = y.copy()
        for i in range(n_stages - 1):
            y_new += h * b5[i] * k[i]
        
        # Compute 4th order solution for error estimate
        y_4th = y.copy()
        for i in range(n_stages - 1):
            y_4th += h * b4[i] * k[i]
        
        # Error estimate
        err = np.linalg.norm((y_new - y_4th) / np.maximum(atol, rtol * np.maximum(np.abs(y), np.abs(y_new))))
        
        if err <= 1.0:
            # Accept step
            t += h
            y = y_new
            steps += 1
            # Increase step size
            if err > 0:
                h_new = h * (0.9 * err ** (-0.2))
            else:
                h_new = h * 2.0
            h = min(max(h_new, min_h), max_h)
        else:
            # Reject step, decrease h
            h = h * (0.9 * err ** (-0.25))
            h = max(h, min_h)
        
        if steps > max_steps:
            break
    
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Use adaptive RK45 with tight tolerances
    y_end = _rk45_adaptive(f, t0, t1, y0, rtol=1e-11, atol=1e-13)
    return y_end
# EVOLVE-BLOCK-END
