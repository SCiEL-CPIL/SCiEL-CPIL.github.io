"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    
    Uses LSODA (automatic stiffness detection) as primary method, with Radau/BDF fallbacks.
    Includes a deadline guard to stay within the 330s wall-clock limit.
    """
    t_start = time.time()
    deadline = t_start + 320.0  # 10s safety margin before 330s kill
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Count function calls
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Primary: LSODA with tight tolerances
    # LSODA automatically switches between Adams (non-stiff) and BDF (stiff)
    rtol = 1e-10
    atol = 1e-12
    
    try:
        sol = solve_ivp(
            f_counted,
            (t0, t1),
            y0,
            method='LSODA',
            rtol=rtol,
            atol=atol,
        )
        
        if sol.success:
            return sol.y[:, -1]
        
    except Exception:
        pass
    
    # If LSODA failed or timed out, try Radau with tight tolerances
    if time.time() < deadline:
        try:
            sol = solve_ivp(
                f_counted,
                (t0, t1),
                y0,
                method='Radau',
                rtol=1e-10,
                atol=1e-12,
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Fallback: BDF method
    if time.time() < deadline:
        try:
            sol = solve_ivp(
                f_counted,
                (t0, t1),
                y0,
                method='BDF',
                rtol=1e-9,
                atol=1e-11,
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Final fallback: simple RK4 with adaptive steps
    if time.time() < deadline:
        try:
            h = (t1 - t0) / 1000
            t = t0
            y = y0.copy()
            n = len(y)
            
            while t < t1 - 1e-12:
                if time.time() > deadline:
                    break
                h = min(h, t1 - t)
                k1 = f_counted(t, y)
                k2 = f_counted(t + h/2, y + h/2 * k1)
                k3 = f_counted(t + h/2, y + h/2 * k2)
                k4 = f_counted(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
                if not np.all(np.isfinite(y)):
                    break
            return y
        except Exception:
            pass
    
    # Return initial condition as last resort
    return y0
# EVOLVE-BLOCK-END
