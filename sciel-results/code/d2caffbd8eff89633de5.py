import numpy as np
import time

def get_dop853_coefficients():
    # This function attempts to extract coefficients from scipy's internal DOP853 implementation
    # to avoid hardcoding errors.
    try:
        from scipy.integrate._ivp.ivp import DOP853
        # Create a dummy instance
        # The constructor signature might vary, but usually it's (f, t_span, y0)
        # We can just access the class attributes if they exist, or instantiate with dummy args.
        # DOP853 is a class in _ivp.ivp
        # Let's try to instantiate it with a dummy function to get the tableau.
        dummy_f = lambda t, y: y
        stepper = DOP853(dummy_f, [0, 1], np.zeros(3))
        return stepper.c, stepper.A, stepper.b, stepper.b_star
    except Exception as e:
        # Fallback to hardcoded values if extraction fails
        # DOP853 coefficients are complex. 
        # If this fails, we fall back to solve_ivp.
        return None

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Try to get coefficients
    coeffs = get_dop853_coefficients()
    
    if coeffs is None:
        # Fallback to standard solve_ivp if we can't get coefficients
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-11, atol=1e-13)
        return sol.y[:, -1]
    
    c, A, b, b_star = coeffs
    n_stages = len(c)
    
    # Determine fixed step size
    # DOP853 is 12th order. 
    # We want to match rtol=1e-11 accuracy.
    # Let's start with h=0.01.
    h = 0.01
    N = int(np.ceil((t1 - t0) / h))
    h = (t1 - t0) / N
    
    y = y0.copy()
    t = t0
    
    # Pre-allocate stage array
    k = np.zeros((n_stages, len(y)))
    
    for i in range(N):
        # Calculate stages
        for s in range(n_stages):
            k[s] = f(t + c[s] * h, y + h * (A[s] @ k[:s]))
        
        # Update y using the 12th order weights 'b'
        # Note: 'b' has length n_stages
        y = y + h * (b @ k)
        t = t + h
        
    return y
