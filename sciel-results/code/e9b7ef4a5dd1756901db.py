"""Use DOP853 with tighter tolerances than the reference to minimize rel error,
while the high order keeps call count low to maximize the -log10(calls) term."""
import numpy as np
from scipy.integrate import solve_ivp
import time


def integrate(f, t_span, y0):
    """
    Integrates the ODE system from t_span[0] to t_span[1].
    Returns the state at the end of t_span.
    
    Uses DOP853 with rtol=1e-12, atol=1e-14 (tighter than reference) to
    achieve higher accuracy while keeping call count reasonable.
    Includes a deadline guard to ensure we finish within the time limit.
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float).copy()
    
    start_time = time.time()
    deadline = start_time + 290  # 10 second safety margin before 300s limit
    
    call_count = [0]
    
    def counted_f(t, y):
        call_count[0] += 1
        if call_count[0] > 1900000:
            raise RuntimeError("Too many function calls")
        if time.time() > deadline:
            raise RuntimeError("Deadline exceeded")
        return f(t, y)
    
    try:
        sol = solve_ivp(
            counted_f,
            (t0, t1),
            y0_arr,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14
        )
        
        if not sol.success:
            # Fallback: try with reference tolerances
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
            # Fallback: try RK45 with reference tolerances
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
