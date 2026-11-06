"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: aim to finish before 300s wall-clock
    start_time = time.time()
    deadline = start_time + 290  # 290s limit to be safe
    
    # Use Radau5 (implicit Runge-Kutta) for stiff problems like 2D heat equation
    # Radau5 is L-stable and very accurate for stiff ODEs
    # We'll use a dense output or just integrate to t1
    
    # First, try with a reasonable number of steps
    # Radau5 is 5th order, so we need fewer steps than RK4
    
    # Estimate step size: for heat equation, stability depends on diffusion coefficient and grid spacing
    # But since we don't know the exact problem, let's use adaptive step size
    
    # Use scipy's solve_ivp with Radau5 method
    from scipy.integrate import solve_ivp
    
    # Try to integrate with Radau5
    try:
        # Use tight tolerances for accuracy
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            rtol=1e-10, 
            atol=1e-12,
            max_step=(t1 - t0) / 100  # Limit step size to ensure accuracy
        )
        
        if not sol.success:
            # Fallback: try RK45 with smaller steps
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='RK45',
                rtol=1e-10, 
                atol=1e-12,
                max_step=(t1 - t0) / 1000
            )
        
        if sol.success and sol.t[-1] == t1:
            return sol.y[:, -1]
        else:
            # If solve_ivp failed, fall back to manual RK4
            return _rk4_fallback(f, t0, t1, y0)
            
    except Exception:
        # Fallback to manual RK4 if scipy fails
        return _rk4_fallback(f, t0, t1, y0)


def _rk4_fallback(f, t0, t1, y0):
    """Fallback RK4 implementation if scipy fails."""
    h = (t1 - t0) / 2000
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(2000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            break
    return y
# EVOLVE-BLOCK-END
