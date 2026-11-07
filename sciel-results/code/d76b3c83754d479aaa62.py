# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the harmonic oscillator ODE: y0' = y1, y1' = -y0
    over t_span with initial condition y0.
    
    Uses DOP853 (8th order explicit Runge-Kutta) with tight tolerances
    to match the reference solution closely while minimizing function calls.
    """
    start_time = time.time()
    deadline = 310.0  # Safety margin before 330s limit
    
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Try DOP853 with same tolerances as reference
    try:
        import scipy.integrate
        sol = scipy.integrate.solve_ivp(
            f,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13
        )
        
        elapsed = time.time() - start_time
        if elapsed < deadline and sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)) and y_end.shape == y0.shape:
                return y_end
    except Exception:
        pass
    
    # Fallback: fixed-step RK4 with adaptive step refinement
    # Start with a reasonable number of steps and increase if needed
    best_y = None
    for n_steps in [1000, 5000, 20000, 100000]:
        if time.time() - start_time > deadline:
            break
        try:
            h = (t1 - t0) / n_steps
            t, y = t0, y0.copy()
            for _ in range(n_steps):
                k1 = f(t, y)
                k2 = f(t + h / 2, y + h / 2 * k1)
                k3 = f(t + h / 2, y + h / 2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
                if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
                    break
            else:
                # Completed all steps
                if np.all(np.isfinite(y)):
                    best_y = y
        except Exception:
            continue
    
    if best_y is not None:
        return best_y
    
    # Last resort: return initial condition (should not happen)
    return y0
# EVOLVE-BLOCK-END
