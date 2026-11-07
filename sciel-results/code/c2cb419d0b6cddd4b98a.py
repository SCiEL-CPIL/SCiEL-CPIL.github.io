import numpy as np
import scipy.integrate

def integrate(f, t_span, y0):
    """
    ODE integrator for the Lorenz system.
    Uses scipy.integrate.odeint (LSODA) for efficiency via multistep methods.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Attempt to use LSODA (odeint)
    # LSODA is a multistep method that can be more efficient in terms of function calls
    # for non-stiff problems because it uses history.
    try:
        # Provide a range of t points to ensure the solver has a good trajectory
        # But we only need the last one.
        # Using just [t0, t1] might be too coarse for the internal adaptive controller
        # to start well, but LSODA handles this.
        t_eval = np.array([t0, t1])
        
        sol = scipy.integrate.odeint(
            f, 
            y0, 
            t_eval, 
            args=(), 
            mxstep=10000, 
            rtol=1e-11, 
            atol=1e-13
        )
        
        y_end = sol[-1]
        if np.all(np.isfinite(y_end)):
            return y_end
            
    except Exception:
        pass

    # Fallback to high-order single-step method
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13
        )
        if sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Last resort: RK4
    steps = 1000
    h = (t1 - t0) / steps
    t = t0
    y = y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y
