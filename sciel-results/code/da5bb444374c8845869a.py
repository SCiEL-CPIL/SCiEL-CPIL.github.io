import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Integrates the ODE using scipy.integrate.solve_ivp with Radau method.
    """
    y0 = np.array(y0, dtype=float)
    t0, t1 = t_span
    
    call_count = 0
    max_calls = 1900000
    start_time = time.time()
    deadline = start_time + 280.0

    def f_counted(t, y):
        nonlocal call_count
        call_count += 1
        if call_count > max_calls:
            raise RuntimeError("Max calls exceeded")
        if time.time() > deadline:
            raise RuntimeError("Time limit exceeded")
        return f(t, y)

    # Use Radau method with tight tolerances
    # Reference uses rtol=1e-11, atol=1e-13
    # We use tighter to minimize rel error
    sol = scipy.integrate.solve_ivp(
        f_counted, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-13, 
        atol=1e-15,
        max_step=0.1 # Limit step size to ensure stability and accuracy, though Radau handles stiffness well
    )
    
    if not sol.success:
        # Fallback if it fails, though unlikely
        # Return last known state or raise
        raise RuntimeError("Integration failed: " + sol.message)

    return sol.y[:, -1]
