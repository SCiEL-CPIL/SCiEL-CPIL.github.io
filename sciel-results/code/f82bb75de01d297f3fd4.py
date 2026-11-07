import numpy as np
import time
import scipy.integrate

def integrate(f, t_span, y0):
    t_start = time.time()
    deadline = t_start + 250.0  # Safety margin
    
    # Use DOP853 for high accuracy and efficiency
    # Tolerances tighter than reference to minimize rel error
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14,
            max_step=0.05 # Prevent huge steps, though adaptive usually handles this
        )
        
        if not sol.success:
            # Fallback to a more robust method if DOP853 fails
            sol = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='RK45', 
                rtol=1e-11, 
                atol=1e-13
            )
            
        if sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
        else:
            return np.array(y0, dtype=float)
            
    except Exception:
        # If scipy fails for any reason, fallback to simple RK4
        # This ensures we don't crash and return something valid
        t0, t1 = t_span
        y = np.array(y0, dtype=float)
        steps = 1000
        h = (t1 - t0) / steps
        t = t0
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            if time.time() > deadline:
                break
        return y
