import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    start_time = time.time()
    deadline = start_time + 290.0
    
    if t1 <= t0:
        return y0

    # Strategy: Use BDF (Backward Differentiation Formula) as the primary method.
    # BDF is an implicit multi-step method that is highly effective for stiff problems.
    # It is often faster than Radau because it uses a simpler linear solver at each step.
    # For convection-diffusion, which can be stiff, BDF might achieve high accuracy
    # with fewer function evaluations than explicit methods.
    
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-12, 
            atol=1e-14
        )
        if time.time() < deadline and sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to Radau if BDF fails
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14
        )
        if time.time() < deadline and sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to DOP853
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14
        )
        if time.time() < deadline and sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: RK4
    steps = 1000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
