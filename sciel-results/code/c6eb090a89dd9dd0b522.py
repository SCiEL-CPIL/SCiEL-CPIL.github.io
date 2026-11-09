"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Estimate the linear operator A by probing with basis vectors
    # f(t, y) = A @ y
    A = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i)
        
    # Compute matrix exponential exp(A * dt)
    dt = t1 - t0
    M = sla.expm(A * dt)
    
    y_end = M @ y0
    return y_end
# EVOLVE-BLOCK-END
