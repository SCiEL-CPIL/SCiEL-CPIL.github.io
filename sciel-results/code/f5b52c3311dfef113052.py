"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _rk45_step(f, t, y, h):
    """
    Single step of the 4th-order Runge-Kutta-Fehlberg method.
    Returns y_new (5th order) and y_new4 (4th order) for error estimation.
    """
    a1 = 0.0
    a2 = 1.0 / 4.0
    a3 = 3.0 / 8.0
    a4 = 12.0 / 13.0
    a5 = 1.0
    a6 = 1.0

    b11 = 1.0 / 4.0
    b21 = 3.0 / 32.0
    b22 = 9.0 / 32.0
    b31 = 1932.0 / 2197.0
    b32 = -7200.0 / 2197.0
    b33 = 7296.0 / 2197.0
    b41 = 439.0 / 216.0
    b42 = -8.0
    b43 = 3680.0 / 513.0
    b44 = -845.0 / 4104.0
    b51 = -8.0 / 27.0
    b52 = 2.0
    b53 = -3544.0 / 2565.0
    b54 = 1859.0 / 4104.0
    b55 = -11.0 / 40.0
    b61 = 16.0 / 135.0
    b62 = 0.0
    b63 = 6656.0 / 12825.0
    b64 = 28561.0 / 56430.0
    b65 = -9.0 / 50.0
    b66 = 2.0 / 55.0

    c = np.array([b11, b21, b22, b31, b32, b33, b41, b42, b43, b44,
                  b51, b52, b53, b54, b55, b61, b62, b63, b64, b65, b66])

    k1 = f(t, y)
    k2 = f(t + a2 * h, y + h * (b21 * k1))
    k3 = f(t + a3 * h, y + h * (b31 * k1 + b32 * k2))
    k4 = f(t + a4 * h, y + h * (b41 * k1 + b42 * k2 + b43 * k3))
    k5 = f(t + a5 * h, y + h * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4))
    k6 = f(t + a6 * h, y + h * (b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5))

    # 5th order solution
    y5 = y + h * (b61 * k1 + b63 * k3 + b64 * k4 + b65 * k5 + b66 * k6)
    
    # 4th order solution (for error estimation)
    y4 = y + h * (2825.0 / 27648.0 * k1 + 18575.0 / 48384.0 * k3 + 13525.0 / 55296.0 * k4 + 277.0 / 14336.0 * k5 + 1.0 / 4.0 * k6)
    
    return y5, y4


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Initial step size
    h = (t1 - t0) / 1000.0
    
    # Tolerance for error control
    rtol = 1e-8
    atol = 1e-10
    
    max_steps = 1000000
    step_count = 0
    
    while t < t1 and step_count < max_steps:
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        y5, y4 = _rk45_step(f, t, y, h)
        
        # Check for non-finite values
        if not np.all(np.isfinite(y5)):
            # Reduce step and try again, or fail
            h *= 0.5
            continue
        
        # Error estimate
        err = np.linalg.norm(y5 - y4)
        scale = np.linalg.norm(atol + rtol * np.maximum(np.abs(y4), np.abs(y5)))
        
        if err <= scale:
            # Accept step
            y = y5
            t += h
            
            # Adjust step size
            if err > 0:
                factor = min(5.0, max(0.1, 0.9 * (scale / err) ** 0.2))
                h *= factor
            else:
                h *= 2.0
        else:
            # Reject step, reduce h
            if err > 0:
                factor = max(0.1, 0.9 * (scale / err) ** 0.25)
                h *= factor
            else:
                h *= 0.5
        
        step_count += 1
    
    return y
# EVOLVE-BLOCK-END
