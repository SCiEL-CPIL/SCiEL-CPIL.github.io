"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system using DOP853.
    Tolerances are tuned to balance accuracy (rel) and efficiency (calls)
    to maximize the score: -log10(rel) - 0.5*log10(calls).
    """
    t0, t1 = t_span
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit

    # Strategy: Use moderately tight tolerances to keep 'rel' small enough
    # (likely below the 1e-14 cap or close to it) while minimizing 'calls'.
    # If previous rtol=1e-11 resulted in score ~1.6, it implies 'rel' might
    # be larger than 1e-14 (e.g., 1e-3) or 'calls' were high.
    # Loosening tolerances reduces 'calls' significantly.
    # We try rtol=1e-7, atol=1e-9.
    
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-7, 
            atol=1e-9,
            t_eval=[t1]
        )
        
        if not sol.success:
            # If it fails, try slightly tighter or different method
            sol = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='RK45', 
                rtol=1e-6, 
                atol=1e-8,
                t_eval=[t1]
            )
            if not sol.success:
                return np.array(y0, dtype=float)

        y_end = sol.y[:, -1]
        
        # Check if result is finite
        if not np.all(np.isfinite(y_end)):
            return np.array(y0, dtype=float)
            
        return y_end
        
    except Exception:
        # Fallback to a simple RK4 if scipy fails entirely
        try:
            y = np.array(y0, dtype=float)
            t = t0
            h = (t1 - t0) / 1000  # 1000 steps
            for _ in range(1000):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
                if time.time() > deadline:
                    break
            return y
        except Exception:
            return np.array(y0, dtype=float)

# EVOLVE-BLOCK-END
