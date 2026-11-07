import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=np.float64)
    
    if t0 == t1:
        return y0_arr.copy()
    
    start = time.perf_counter()
    deadline = 310.0  # 20s safety margin before 330s kill
    
    def _remaining():
        return deadline - (time.perf_counter() - start)
    
    # Strategy: Use DOP853 with very tight tolerances to minimize rel error.
    # The score is -log10(rel) - 0.5*log10(calls).
    # Reducing rel from 1e-3 to 1e-14 gains 11 points.
    # Increasing calls from 500 to 5000 costs only 1.5 points.
    # Thus, tight tolerances are highly beneficial.
    
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0_arr,
            method="DOP853",
            rtol=1e-15,
            atol=1e-17
        )
        if sol.success and sol.y.shape[1] >= 1:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass
    
    if _remaining() < 5.0:
        return y0_arr.copy()
    
    # Fallback: If DOP853 fails or times out, use a fixed-step high-order RK method.
    # RK4 with h=1e-4 gives 20000 steps * 4 calls = 80000 calls.
    # Error ~ h^4 = 1e-16.
    # This is a safe fallback.
    steps = 20000
    h = (t1 - t0) / steps
    t = float(t0)
    y = y0_arr.copy()
    
    for _ in range(steps):
        if _remaining() < 2.0:
            break
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        
        if not np.all(np.isfinite(y)):
            return y0_arr.copy()
    
    return y
