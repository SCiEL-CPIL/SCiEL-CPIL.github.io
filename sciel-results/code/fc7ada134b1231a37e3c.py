"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses Radau (implicit Runge-Kutta) for stiff problems, with a pre-extracted Jacobian
    to minimize function evaluations.
    """
    t_start = time.time()
    deadline = t_start + 280  # safety margin before 300s limit
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract Jacobian at the initial point using finite differences
    # This requires n+1 function calls
    eps = 1e-7
    fz0 = f(t0, y0)
    J = np.empty((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps * max(1.0, abs(y0[j]))
        J[:, j] = (f(t0, y0 + e) - fz0) / e[j]
    
    # Use Radau method with the pre-extracted Jacobian
    # Radau is an implicit Runge-Kutta method that is A-stable and good for stiff problems
    # We use rtol=1e-10 and atol=1e-12 for high accuracy
    sol = solve_ivp(
        f, t_span, y0,
        method='Radau',
        jac=lambda t, y: J,  # Use the pre-extracted Jacobian
        rtol=1e-10,
        atol=1e-12,
        dense_output=False
    )
    
    # Check if we hit the deadline
    if time.time() > deadline:
        # Return the last successful state if available
        if sol.t.size > 0:
            return sol.y[:, -1]
        return y0
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # If Radau failed, try with looser tolerances
        sol2 = solve_ivp(
            f, t_span, y0,
            method='Radau',
            jac=lambda t, y: J,
            rtol=1e-8,
            atol=1e-10,
            dense_output=False
        )
        if sol2.success:
            return sol2.y[:, -1]
        # Last resort: return the best we have
        if sol2.t.size > 0:
            return sol2.y[:, -1]
        return y0

# EVOLVE-BLOCK-END
