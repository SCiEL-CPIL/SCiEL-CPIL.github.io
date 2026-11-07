import time
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    y0 = np.array(y0, dtype=float)
    
    # Primary strategy: High-order adaptive solver
    # This minimizes f calls while maintaining high accuracy
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15
        )
    except Exception:
        # Fallback if DOP853 has issues
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='RK45',
                rtol=1e-11,
                atol=1e-13
            )
        except Exception:
            # Last resort: Fixed step RK4
            # This is slow but robust
            t0, t1 = t_span
            h = (t1 - t0) / 1000
            t, y = t0, y0.copy()
            for _ in range(1000):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
                if time.time() > deadline:
                    return y
            return y

    if not sol.success:
        # If solve_ivp failed, try a simple fixed step method as fallback
        t0, t1 = t_span
        h = (t1 - t0) / 1000
        t, y = t0, y0.copy()
        for _ in range(1000):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            if time.time() > deadline:
                return y
        return y

    return sol.y[:, -1]
