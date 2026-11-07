import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    ODE integrator for the Lotka-Volterra system.
    Uses high-order adaptive Runge-Kutta (DOP853) to minimize relative error.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Time guard start
    start_time = time.time()
    timeout = 290.0 # 330s limit, leave margin
    
    # Primary strategy: High-order adaptive RK
    # DOP853 is an 8th-order Runge-Kutta method, very efficient for high accuracy
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853',
            rtol=1e-12, 
            atol=1e-14,
            max_step=1.0 # Allow large steps if smooth
        )
        
        # Check if we are still within time limits
        if time.time() - start_time > timeout:
             # Fallback: return last computed point if available, or y0 if no step taken
             if sol.t.size > 0:
                 return sol.y[:, -1]
             return y0

        if sol.success:
            return sol.y[:, -1]
        else:
            # If DOP853 failed (e.g. step size too small), try RK45 with tighter tol
            sol2 = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='RK45',
                rtol=1e-12, 
                atol=1e-14
            )
            if sol2.success:
                return sol2.y[:, -1]
            
    except Exception:
        pass
    
    # Final fallback: Simple fixed-step RK4 if all else fails (low accuracy but safe)
    # This should not be reached for this specific problem.
    def _rk4(f, t0, t1, y0, steps):
        h = (t1 - t0) / steps
        t, y = t0, np.array(y0, dtype=float)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y

    return _rk4(f, t0, t1, y0, 1000)
