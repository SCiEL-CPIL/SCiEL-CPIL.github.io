"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin from 330s limit

    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        # Fallback to manual RK45
        def _rk45(f, t0, t1, y0, rtol=1e-10, atol=1e-12):
            y = np.array(y0, dtype=float)
            t = t0
            h = (t1 - t0) / 100.0
            while t < t1:
                h = min(h, t1 - t)
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2*k1)
                k3 = f(t + h/2, y + h/2*k2)
                k4 = f(t + h, y + h*k3)
                k5 = f(t + h/5, y + h/5*k4)
                k6 = f(t + 3*h/5, y + 3*h/5*(k1 + 4/3*(k4-k1)))
                
                y_new = y + h*(k1 + 3*k2 + 3*k3 + k4)/8
                err = h*(k4 - k3)/32
                err_norm = np.max(np.abs(err) / (1 + np.abs(y_new)))
                
                if err_norm <= rtol:
                    y = y_new
                    t += h
                    h *= min(5.0, 0.9*(rtol/err_norm)**0.2) if err_norm > 0 else 5.0
                else:
                    h *= max(0.1, 0.9*(rtol/err_norm)**0.2)
                if time.time() > deadline:
                    break
            return y
        
        return _rk45(f, t0, t1, y0)

    # Primary: Radau with very tight tolerances, no max_step, small first step
    # Radau is L-stable, excellent for stiff problems like Robertson
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-12, atol=1e-14,
            first_step=1e-8
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback 1: Radau without first_step hint
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-12, atol=1e-14
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback 2: BDF with tight tolerances
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-12, atol=1e-14
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback 3: LSODA
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='LSODA',
            rtol=1e-12, atol=1e-14
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: RK45
    def _rk45_fallback(f, t0, t1, y0, rtol=1e-10, atol=1e-12):
        y = np.array(y0, dtype=float)
        t = t0
        h = (t1 - t0) / 100.0
        while t < t1 and time.time() < deadline:
            h = min(h, t1 - t)
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            k5 = f(t + h/5, y + h/5*k4)
            k6 = f(t + 3*h/5, y + 3*h/5*(k1 + 4/3*(k4-k1)))
            
            y_new = y + h*(k1 + 3*k2 + 3*k3 + k4)/8
            err = h*(k4 - k3)/32
            err_norm = np.max(np.abs(err) / (1 + np.abs(y_new)))
            
            if err_norm <= rtol:
                y = y_new
                t += h
                h *= min(5.0, 0.9*(rtol/err_norm)**0.2) if err_norm > 0 else 5.0
            else:
                h *= max(0.1, 0.9*(rtol/err_norm)**0.2)
        return y
    
    return _rk45_fallback(f, t0, t1, y0)
# EVOLVE-BLOCK-END
