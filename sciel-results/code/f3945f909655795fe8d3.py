"""ODE integrator for the harmonic oscillator using high-order Taylor series."""
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the ODE over t_span using a 16th-order Taylor series method.
    This approach minimizes function calls by using a high-order method
    with a large fixed step size, exploiting the smoothness and linearity
    of the harmonic oscillator.
    """
    y = np.asarray(y0, dtype=float)
    t_start, t_end = t_span
    T = t_end - t_start
    
    if T == 0:
        return y

    # Order of Taylor series
    P = 16
    
    # Estimate step size h such that truncation error is ~ 1e-14
    # Error ~ h^P / P!
    # 16! ~ 2.09e13
    # We want h^P / P! <= 1e-14
    # h <= (1e-14 * 2.09e13)^(1/16) ~ 0.91
    # Use h = 0.9 to be safe
    h = 0.9
    n_steps = int(np.ceil(T / h))
    # Adjust h to exactly cover T
    h = T / n_steps
    
    # Precompute coefficients h^k / k!
    h_pows = np.zeros(P + 1)
    h_pows[0] = 1.0
    for k in range(1, P + 1):
        h_pows[k] = h_pows[k-1] * h / k
        
    t = t_start
    for _ in range(n_steps):
        y_new = np.zeros_like(y)
        v = y
        y_new += h_pows[0] * v
        
        for k in range(1, P + 1):
            v = f(t, v)
            y_new += h_pows[k] * v
            
        y = y_new
        t += h
        
    return y
