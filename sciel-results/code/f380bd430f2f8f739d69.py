"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Uses adaptive high-order Runge-Kutta (DOP853) for efficiency and accuracy.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Deadline guard: keep margin before 300s limit
    start_time = time.time()
    deadline = 280.0  # seconds
    
    # First pass: DOP853 with tight tolerances
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0_arr,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 100.0  # reasonable max step to ensure resolution
        )
        
        if sol.success and sol.t[-1] >= t1 - 1e-15:
            y_final = sol.y[:, -1]
            
            # Check if we're still within time budget for a potential refinement
            elapsed = time.time() - start_time
            if elapsed < deadline - 30.0:
                # Try an even tighter tolerance if we have time
                sol2 = scipy.integrate.solve_ivp(
                    f,
                    t_span,
                    y0_arr,
                    method='DOP853',
                    rtol=1e-14,
                    atol=1e-16,
                    max_step=(t1 - t0) / 200.0
                )
                if sol2.success and sol2.t[-1] >= t1 - 1e-15:
                    y_final = sol2.y[:, -1]
            
            return y_final
        else:
            # If solve_ivp didn't complete, fall back to RK4
            pass
            
    except Exception:
        pass
    
    # Fallback: fixed-step RK4
    h = (t1 - t0) / 5000.0
    t = t0
    y = y0_arr.copy()
    n_steps = int(round((t1 - t0) / h))
    
    for _ in range(n_steps):
        if time.time() - start_time > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            return y
    
    return y
# EVOLVE-BLOCK-END
