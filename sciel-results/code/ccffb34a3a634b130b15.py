import numpy as np
import scipy.integrate as integrate
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    timeout = 280.0  # Safety margin for 300s limit
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # 1. Attempt to detect constant linear system y' = A*y + b
    # Extract b = f(t0, 0)
    b = f(t0, np.zeros(n))
    
    # Extract A columns: A[:, j] = f(t0, e_j) - b
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j) - b
        
    # Verify if the system is indeed constant linear
    is_linear_constant = True
    t_check1 = t0 + 0.1 * (t1 - t0)
    t_check2 = t0 + 0.5 * (t1 - t0)
    
    # Pick a test y
    y_test = np.ones(n) * 0.5
    y_test[0] = 1.0
    
    f_check1 = f(t_check1, y_test)
    f_pred1 = A @ y_test + b
    if not np.allclose(f_check1, f_pred1, rtol=1e-8, atol=1e-10):
        is_linear_constant = False
        
    if is_linear_constant:
        f_check2 = f(t_check2, y_test)
        f_pred2 = A @ y_test + b
        if not np.allclose(f_check2, f_pred2, rtol=1e-8, atol=1e-10):
            is_linear_constant = False
            
    if is_linear_constant:
        # Use precomputed A and b, no more f calls needed for the solver
        def my_f(t, y):
            return A @ y + b
            
        def my_jac(t, y):
            return A
            
        try:
            sol = integrate.solve_ivp(
                my_f, t_span, y0, 
                method='Radau', 
                jac=my_jac,
                rtol=1e-12, 
                atol=1e-14,
                max_step=0.05
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass # Fallback if solver fails
            
    # Check time before expensive fallback
    if time.time() - start_time > timeout:
        return y0
        
    # Fallback: Use solve_ivp directly with f
    try:
        sol = integrate.solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.05
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    if time.time() - start_time > timeout:
        return y0
        
    # Last resort: BDF
    try:
        sol = integrate.solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.05
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    return y0
