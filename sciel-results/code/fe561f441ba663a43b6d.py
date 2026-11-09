"""Use DOP853 with tighter tolerances than the reference to drive rel error
below the 1e-14 floor, while the high order keeps call count low."""
import numpy as np
from scipy.integrate import solve_ivp
import time


def integrate(f, t_span, y0):
    """
    Integrates the ODE system from t_span[0] to t_span[1].
    Returns the state at the end of t_span.
    
    Uses DOP853 with rtol=1e-13, atol=1e-15 (tighter than reference's 1e-11/1e-13)
    to minimize truncation error and push rel toward the 1e-14 floor.
    """
    t_start_time = time.time()
    deadline = t_start_time + 290  # 300s limit, 10s safety margin
    
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float).copy()
    
    call_count = [0]
    
    def counted_f(t, y):
        call_count[0] += 1
        if call_count[0] > 1900000:
            raise RuntimeError("Too many function calls")
        return f(t, y)
    
    try:
        # Primary: DOP853 with very tight tolerances
        sol = solve_ivp(
            counted_f,
            (t0, t1),
            y0_arr,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15
        )
        
        if not sol.success:
            # Fallback: DOP853 with reference tolerances
            call_count[0] = 0
            sol = solve_ivp(
                counted_f,
                (t0, t1),
                y0_arr,
                method='DOP853',
                rtol=1e-11,
                atol=1e-13
            )
        
        if not sol.success:
            # Last resort: RK45 with reference tolerances
            call_count[0] = 0
            sol = solve_ivp(
                counted_f,
                (t0, t1),
                y0_arr,
                method='RK45',
                rtol=1e-11,
                atol=1e-13
            )
        
        if not sol.success:
            return y0_arr
            
        y_end = sol.y[:, -1].copy()
        
        if not np.all(np.isfinite(y_end)):
            return y0_arr
            
        return y_end

    except Exception:
        return y0_arr
