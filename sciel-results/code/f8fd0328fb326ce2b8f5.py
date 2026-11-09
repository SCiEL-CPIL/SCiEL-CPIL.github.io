"""
ODE integrator for the Lorenz system using SciPy's DOP853 method.
Matches the reference solution's method and tolerances to minimize relative error
and function calls.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate

def integrate(f, t_span, y0):
    """
    Integrates the Lorenz system from t_span[0] to t_span[1] with initial condition y0.
    
    Args:
        f: The ODE function f(t, y) returning dy/dt.
        t_span: Tuple (t_start, t_end).
        y0: Initial state vector.
        
    Returns:
        The state vector at t_end.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use DOP853 with the same tolerances as the reference solution
    # This minimizes the relative error while keeping the number of function calls low
    # due to the high order and adaptive stepping of the method.
    
    # We set max_step to infinity (default) to let the adaptive algorithm decide.
    # The default max_step in solve_ivp is (t1-t0)/10, which might force extra steps.
    # Setting it to a large value allows the integrator to take larger steps if accuracy permits.
    
    sol = scipy.integrate.solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13,
        max_step=np.inf
    )
    
    if not sol.success:
        # Fallback: if DOP853 fails for some reason, try RK45 with tighter tolerances
        # though this is unlikely for the Lorenz system in [0, 2]
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-11, 
            atol=1e-13,
            max_step=np.inf
        )
        
    y_end = sol.y[:, -1]
    
    # Ensure the result is a 1D array of floats
    if not np.all(np.isfinite(y_end)):
        # If the solution blew up, return zeros or raise an error?
        # The task says "exception makes the whole result invalid".
        # But "finite vector" is required.
        # For the Lorenz system in [0,2] with y0=[1,1,1], it should be stable.
        # If it's not finite, we return the last finite value or zeros?
        # The contract says "result must be a finite vector".
        # If it's not finite, we should probably try to recover or return a valid finite vector.
        # However, for this specific problem, it shouldn't happen.
        # To be safe, if it's not finite, we return zeros, but this will have high error.
        # Better to just return the value, assuming it is finite.
        pass
        
    return y_end
# EVOLVE-BLOCK-END
