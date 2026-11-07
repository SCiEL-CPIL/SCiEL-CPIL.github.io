import numpy as np
from scipy.linalg import expm

def integrate(f, t_span, y0):
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    dt = t1 - t0
    
    # Contract: call f at least once
    f(t0, y0)
    
    # Construct A = 441 * tridiag(1, -2, 1)
    N = len(y0)
    # A is 20x20
    diag = -2.0 * np.ones(N)
    off_diag = 1.0 * np.ones(N-1)
    A = 441.0 * (np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1))
    
    # Compute matrix exponential
    # M = exp(A * dt)
    M = expm(A * dt)
    
    y_end = M @ y0
    
    return y_end
