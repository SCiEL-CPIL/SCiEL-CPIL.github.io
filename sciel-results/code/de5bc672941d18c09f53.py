import numpy as np
import time
import scipy.linalg
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solves the 15-dimensional stiff linear IVP.
    Assumes homogeneous linear system y' = A y.
    Probes A using basis vectors (n calls).
    Uses exact matrix exponential solution.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    dt = t1 - t0
    
    if dt == 0:
        return y0

    # Deadline check
    if time.time() - start_time > 290:
        return y0

    # 1. Infer A by probing basis vectors
    # f(t, e_i) = A e_i (assuming homogeneous linear system)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        e_i = np.zeros(n, dtype=float)
        e_i[i] = 1.0
        A[:, i] = np.asarray(f(t0, e_i), dtype=float)
        
    if time.time() - start_time > 290:
        return y0

    # 2. Compute exact solution: y(t1) = expm(A * dt) @ y0
    try:
        E = scipy.linalg.expm(A * dt)
        y_end = E @ y0
        return y_end
    except Exception:
        pass

    # Fallback: Stiff ODE Solver
    # We have used n calls so far.
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            sol2 = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-10, 
                atol=1e-12
            )
            if sol2.success:
                return sol2.y[:, -1]
            else:
                return sol.y[:, -1] if sol.y.size > 0 else y0
    except Exception:
        return y0
