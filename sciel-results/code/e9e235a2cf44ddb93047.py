import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Solves the Robertson ODE problem using an adaptive implicit solver.
    """
    # Wrap f to count evaluations
    call_count = 0
    
    def f_counted(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)
    
    # Analytic Jacobian for the Robertson problem
    # y = [y1, y2, y3]
    # y1' = -0.04*y1 + 1e4*y2*y3
    # y2' = 0.04*y1 - 1e4*y2*y3 - 6e3*y2^2
    # y3' = 6e3*y2^2
    def robertson_jac(t, y):
        y1, y2, y3 = y[0], y[1], y[2]
        J = np.array([
            [-0.04, 1e4 * y3, 1e4 * y2],
            [0.04, -1e4 * y3 - 6e3 * y2, -1e4 * y2],
            [0.0, 6e3 * y2, 0.0]
        ])
        return J

    start_time = time.time()
    deadline = 320.0

    # Strategy: Use Radau with very tight tolerances.
    # Radau is generally more accurate for stiff problems like Robertson than BDF
    # at the same number of steps, or achieves the same accuracy with fewer steps/calls.
    # The score is -log10(rel_err) - 0.5*log10(calls).
    # Tighter tolerance reduces rel_err (increases first term) but increases calls (decreases second term).
    # Since the penalty for calls is only 0.5 * log10, we can afford to spend more calls to get much lower error.
    
    try:
        # First attempt: Radau with tight tolerances
        sol = solve_ivp(
            f_counted, 
            t_span, 
            y0, 
            method='Radau', 
            jac=robertson_jac,
            rtol=1e-13, 
            atol=1e-15,
            vectorized=False
        )
        
        if not sol.success:
            # Fallback to BDF if Radau fails (shouldn't happen for Robertson)
            sol = solve_ivp(
                f_counted, 
                t_span, 
                y0, 
                method='BDF', 
                jac=robertson_jac,
                rtol=1e-13, 
                atol=1e-15,
                vectorized=False
            )
            
        if not sol.success:
            # Last resort: looser tolerances
            sol = solve_ivp(
                f_counted, 
                t_span, 
                y0, 
                method='Radau', 
                jac=robertson_jac,
                rtol=1e-10, 
                atol=1e-12,
                vectorized=False
            )

        return sol.y[:, -1]
        
    except Exception:
        # Safety net: simple RK4 with adaptive step
        t0, tf = t_span
        y = np.array(y0, dtype=float)
        t = t0
        h = 0.01
        while t < tf and (time.time() - start_time) < deadline:
            if t + h > tf:
                h = tf - t
            
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            
            y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Error estimate
            k1a = f(t, y)
            k2a = f(t + h/4, y + h/4 * k1a)
            k3a = f(t + h/4, y + h/4 * k2a)
            k4a = f(t + h/2, y + h/2 * k3a)
            y_half = y + h/2/6 * (k1a + 2*k2a + 2*k3a + k4a)
            
            k1b = f(t + h/2, y_half)
            k2b = f(t + 3*h/4, y_half + h/4 * k1b)
            k3b = f(t + 3*h/4, y_half + h/4 * k2b)
            k4b = f(t + h, y_half + h/2 * k3b)
            y_double = y_half + h/2/6 * (k1b + 2*k2b + 2*k3b + k4b)
            
            err = np.max(np.abs(y_new - y_double))
            y = y_double
            
            if err < 1e-10:
                t += h
                h = min(h * 1.5, 1.0)
            else:
                h = max(h * 0.5, 1e-6)
                
        return y
