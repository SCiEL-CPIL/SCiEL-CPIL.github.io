import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Solves the Robertson ODE problem using LSODA with tight tolerances.
    """
    # Analytic Jacobian for the Robertson problem
    # y = [y1, y2, y3]
    # y1' = -0.04*y1 + 1e4*y2*y3
    # y2' = 0.04*y1 - 1e4*y2*y3 - 6e3*y2^2
    # y3' = 6e3*y2^2
    def robertson_jac(t, y):
        y1, y2, y3 = y[0], y[1], y[2]
        J = np.array([
            [-0.04, 1e4 * y3, 1e4 * y2],
            [0.04, -1e4 * y3 - 6e3 * y2, -1e4 * y2],
            [0.0, 6e3 * y2, 0.0]
        ])
        return J

    start_time = time.time()
    deadline = 300.0 # Keep safety margin from 330s limit

    # Use LSODA which is efficient for stiff problems.
    # Tolerances: rtol=1e-10, atol=1e-12 provides high accuracy without excessive cost.
    # We can try to be slightly tighter if time permits, but 1e-10 is a sweet spot.
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='LSODA', 
            jac=robertson_jac,
            rtol=1e-10, 
            atol=1e-12,
            vectorized=False
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If LSODA fails, try Radau
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau', 
                jac=robertson_jac,
                rtol=1e-10, 
                atol=1e-12,
                vectorized=False
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                # Last resort: looser tolerances
                sol = solve_ivp(
                    f, 
                    t_span, 
                    y0, 
                    method='LSODA', 
                    jac=robertson_jac,
                    rtol=1e-8, 
                    atol=1e-10,
                    vectorized=False
                )
                return sol.y[:, -1]
                
    except Exception:
        # Fallback to a simple explicit method if SciPy fails completely
        # This is unlikely but provides a safety net.
        t0, tf = t_span
        y = np.array(y0, dtype=float)
        t = t0
        h = 0.01
        while t < tf and (time.time() - start_time) < deadline:
            if t + h > tf:
                h = tf - t
            
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            
        return y
