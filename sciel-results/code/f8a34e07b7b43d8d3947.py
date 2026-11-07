"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Ensure f is called at least once
    _ = f(t0, y0_arr)
    
    deadline = time.time() + 300  # safety margin before 330s kill
    
    try:
        from scipy.integrate import solve_ivp
        
        # Use DOP853 with very tight tolerances to drive relative error to the floor.
        # DOP853 is 8th order, so it's very accurate per step.
        # Tight rtol/atol ensures high accuracy.
        # No max_step restriction allows the solver to take large steps where possible, minimizing calls.
        sol = solve_ivp(
            f, 
            (t0, t1), 
            y0_arr, 
            method='DOP853', 
            rtol=1e-14, 
            atol=1e-16, 
            dense_output=False
        )
        
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
        else:
            # If solve_ivp fails, try with slightly looser tolerances or different method
            # But DOP853 is very robust.
            sol = solve_ivp(
                f, 
                (t0, t1), 
                y0_arr, 
                method='DOP853', 
                rtol=1e-12, 
                atol=1e-14, 
                dense_output=False
            )
            if sol.success and sol.y.shape[1] > 0:
                y_end = sol.y[:, -1]
                if np.all(np.isfinite(y_end)):
                    return y_end
                    
    except Exception:
        pass

    # Fallback: High-order fixed-step RK4 with many steps
    # This is a last resort, should rarely be used.
    n_steps = 10000
    h = (t1 - t0) / n_steps
    t = t0
    y = y0_arr.copy()
    
    for _ in range(n_steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y
# EVOLVE-BLOCK-END
