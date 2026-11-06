import numpy as np
from scipy.integrate import ode

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
        
    # Use lsoda (ODEPACK) which is a hybrid stiff/non-stiff solver
    # It is often more efficient than Radau for Robertson
    solver = ode(f_counted, t0, y0)
    solver.set_integrator('lsoda', rtol=1e-12, atol=1e-14)
    
    # Integrate to t1
    success = solver.integrate(t1)
    
    if success:
        return solver.y
    else:
        # Fallback if lsoda fails (unlikely for Robertson)
        # Try Radau via solve_ivp as a backup
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            f_counted,
            t_span,
            y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.1
        )
        if sol.success:
            return sol.y[:, -1]
        return y0
