"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses BDF adaptive implicit solver with tight tolerances for the Robertson problem.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Track elapsed time for deadline guard
    start_time = time.time()
    deadline = start_time + 280  # Leave 50s margin for 330s limit
    
    def time_left():
        return deadline - time.time()
    
    # Primary strategy: BDF with tight tolerances, no max_step restriction
    # BDF is efficient for stiff problems and reuses derivative info well
    try:
        sol = solve_ivp(
            f, (t0, t1), y0_arr,
            method='BDF',
            rtol=1e-14,
            atol=1e-16
        )
        if sol.success:
            result = sol.y[:, -1]
            
            # Polish pass: if we have time, do a short high-accuracy refinement
            # from t1 - delta to t1 using the current state as initial condition
            if time_left() > 30:
                delta = 0.1
                try:
                    sol2 = solve_ivp(
                        f, (t1 - delta, t1), result,
                        method='Radau',
                        rtol=1e-15,
                        atol=1e-17
                    )
                    if sol2.success:
                        return sol2.y[:, -1]
                except Exception:
                    pass
            
            return result
    except Exception:
        pass
    
    # Fallback: BDF with slightly looser tolerances
    if time_left() > 20:
        try:
            sol = solve_ivp(
                f, (t0, t1), y0_arr,
                method='BDF',
                rtol=1e-13,
                atol=1e-15
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Fallback: Radau
    if time_left() > 15:
        try:
            sol = solve_ivp(
                f, (t0, t1), y0_arr,
                method='Radau',
                rtol=1e-13,
                atol=1e-15
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Last resort: RK4 with moderate steps
    if time_left() > 5:
        n_steps = 20000
        h = (t1 - t0) / n_steps
        t, y = t0, y0_arr.copy()
        for i in range(n_steps):
            if time_left() < 3:
                break
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            if not np.all(np.isfinite(y)):
                break
    
    return y
# EVOLVE-BLOCK-END
