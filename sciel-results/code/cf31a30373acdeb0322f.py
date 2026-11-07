import numpy as np
from scipy.integrate import solve_ivp

def _rk4_fallback(f, t0, t1, y0, steps=5000):
    """Fallback RK4 integrator if solve_ivp fails."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            return y
    return y

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    try:
        # Use DOP853 with tight tolerances to match the reference solution
        # This method is high-order and efficient, minimizing function calls.
        sol = solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13
        )
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
        else:
            return _rk4_fallback(f, t0, t1, y0_arr)
    except Exception:
        return _rk4_fallback(f, t0, t1, y0_arr)
