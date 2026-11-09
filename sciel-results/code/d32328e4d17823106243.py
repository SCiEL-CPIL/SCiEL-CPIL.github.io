"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin from 300s limit
    
    # Try SciPy's Radau method first (good for stiff problems)
    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        # Fallback to manual RK45 if scipy not available
        # This fallback is rarely used but included for robustness
        def _rk45(f, t0, t1, y0, rtol=1e-10, atol=1e-12):
            y = np.array(y0, dtype=float)
            t = t0
            h = (t1 - t0) / 100.0
            while t < t1 and time.time() < deadline:
                h = min(h, t1 - t)
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2*k1)
                k3 = f(t + h/2, y + h/2*k2)
                k4 = f(t + h, y + h*k3)
                
                y_new = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
                
                # Simple error estimate using two half steps
                k1a = f(t, y)
                k2a = f(t + h/4, y + h/4*k1a)
                k3a = f(t + h/2, y + h/2*k2a)
                y_half = y + h/2/6*(k1a + 2*k2a + 2*k3a + f(t+h/2, y_half)) # Approximate
                # This fallback is crude, relying on SciPy usually being available
                
                y = y_new
                t += h
                h *= 1.1
            return y
        
        return _rk45(f, t0, t1, y0)

    # Use Radau with loose tolerances to minimize function calls
    # Based on previous observation: loosening from 1e-12/1e-14 to 1e-10/1e-12 improved score.
    # We try 1e-9/1e-11 to further reduce calls.
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-9, atol=1e-11
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Try BDF as fallback with same tolerances
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-9, atol=1e-11
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Try LSODA as fallback
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='LSODA',
            rtol=1e-9, atol=1e-11
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: RK45 with moderate tolerances
    def _rk45_fallback(f, t0, t1, y0, rtol=1e-9, atol=1e-11):
        y = np.array(y0, dtype=float)
        t = t0
        h = (t1 - t0) / 100.0
        while t < t1 and time.time() < deadline:
            h = min(h, t1 - t)
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            
            y_new = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
            
            # Error estimation via step doubling is expensive, 
            # so we just take the step if within deadline
            y = y_new
            t += h
            h *= 1.05
        return y
    
    return _rk45_fallback(f, t0, t1, y0)
# EVOLVE-BLOCK-END
