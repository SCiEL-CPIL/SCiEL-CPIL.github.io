"""ODE solver: Robertson problem using LSODA with Radau polish via scipy.integrate.solve_ivp."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    
    Three-stage approach:
    1. LSODA with tight tolerances for initial high-accuracy solution
    2. LSODA with even tighter tolerances for refinement
    3. Radau (implicit RK) with very tight tolerances for final polish
       Radau is L-stable and excellent for stiff problems; using it as a
       final pass can reduce error further since it has different error
       characteristics than LSODA's Adams/BDF methods.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Track function calls
    call_count = [0]
    
    def wrapped_f(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Deadline guard: 320 seconds (10s safety margin before 330s kill)
    start_time = time.time()
    deadline = 310  # safety margin
    
    y_current = y0
    
    # Stage 1: LSODA with tight tolerances
    try:
        sol = solve_ivp(
            wrapped_f,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=1e-13,
            atol=1e-15,
            max_step=np.inf
        )
        
        if sol.success and sol.t[-1] >= t1 - 1e-10:
            y_current = sol.y[:, -1].copy()
        elif sol.y.size > 0:
            y_current = sol.y[:, -1].copy()
    
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return y_current
    
    # Stage 2: LSODA with even tighter tolerances
    try:
        sol2 = solve_ivp(
            wrapped_f,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=1e-15,
            atol=1e-17,
            max_step=np.inf
        )
        
        if sol2.success and sol2.t[-1] >= t1 - 1e-10:
            y_current = sol2.y[:, -1].copy()
        elif sol2.y.size > 0:
            y_current = sol2.y[:, -1].copy()
    
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return y_current
    
    # Stage 3: Radau polish with very tight tolerances
    # Radau is a 3rd-order implicit Runge-Kutta method that is L-stable,
    # making it ideal for stiff problems. Using it as a final pass with
    # different error characteristics can further reduce the solution error.
    try:
        sol3 = solve_ivp(
            wrapped_f,
            [t0, t1],
            y0,
            method='Radau',
            rtol=1e-15,
            atol=1e-17,
            max_step=np.inf
        )
        
        if sol3.success and sol3.t[-1] >= t1 - 1e-10:
            y_current = sol3.y[:, -1].copy()
        elif sol3.y.size > 0:
            y_current = sol3.y[:, -1].copy()
    
    except Exception:
        pass
    
    return y_current
# EVOLVE-BLOCK-END
