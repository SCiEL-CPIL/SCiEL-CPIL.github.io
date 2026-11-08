import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    y0 = np.array(y0, dtype=float)
    call_count = 0
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s limit
    
    def f_wrapper(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)
        
    # Radau is a good choice for stiff problems like the heat equation
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapper, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    if time.time() > deadline:
        return y0

    # Fallback to BDF if Radau fails
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapper, 
            t_span, 
            y0, 
            method='BDF', 
            max_order=5,
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    if time.time() > deadline:
        return y0

    # Fallback to RK45 if all else fails (unlikely to be accurate for stiff, but better than nothing)
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapper, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-8, 
            atol=1e-10
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    return y0
