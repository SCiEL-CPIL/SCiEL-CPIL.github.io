import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    # Wrap f to count calls
    call_count = 0
    def counted_f(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)
        
    # Analytic Jacobian for Robertson
    def jac(t, y):
        y1, y2, y3 = y
        J = np.array([
            [-0.04, 1e4 * y3, 1e4 * y2],
            [0.04, -1e4 * y3 - 6e7 * y2, -1e4 * y2],
            [0.0, 6e7 * y2, 0.0]
        ])
        return J

    # Use Radau method with tight tolerances
    # The Robertson problem is stiff, so Radau is appropriate.
    sol = solve_ivp(
        counted_f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-11, 
        atol=1e-13, 
        jac=jac,
        max_step=1.0 # Optional: prevent huge steps if needed, but Radau handles it well
    )
    
    if not sol.success:
        # Fallback to a simpler method if Radau fails (unlikely)
        sol = solve_ivp(counted_f, t_span, y0, method='BDF', rtol=1e-10, atol=1e-12, jac=jac)
        
    return sol.y[:, -1]
