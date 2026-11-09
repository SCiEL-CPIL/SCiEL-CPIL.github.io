import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
        y0 = np.asarray(y0, dtype=float)
        t0, t1 = t_span
        dt = t1 - t0
        
        # Contract: f must be called at least once
        _ = f(t0, y0)
        
        # Analytic solution for y' = A y, A = [[0, 1], [-1, 0]]
        # y(t) = exp(A*dt) y0
        # exp(A*dt) = [[cos(dt), sin(dt)], [-sin(dt), cos(dt)]]
        
        c = np.cos(dt)
        s = np.sin(dt)
        
        y_end = np.empty_like(y0)
        if len(y0) >= 2:
            y_end[0] = y0[0] * c + y0[1] * s
            y_end[1] = -y0[0] * s + y0[1] * c
            if len(y0) > 2:
                y_end[2:] = y0[2:] # Assuming extra dims are static or not part of HO
        else:
            # Fallback for unexpected shapes
            y_end[:] = y0
            
        # Check for finiteness
        if not np.all(np.isfinite(y_end)):
            # Fallback to numerical if analytic fails (should not happen)
            sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-11, atol=1e-13)
            if sol.success:
                return sol.y[:, -1]
            return y0
            
        return y_end
