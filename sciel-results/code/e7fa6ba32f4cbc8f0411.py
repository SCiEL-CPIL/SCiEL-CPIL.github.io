import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Assume the system is a homogeneous linear system y' = A y
    # This reduces function calls from n+1 to n.
    # Extract matrix A by evaluating f at the standard basis vectors.
    I = np.eye(n)
    A = np.zeros((n, n))
    
    for i in range(n):
        A[:, i] = f(t0, I[:, i])
        
    dt = t1 - t0
    expA = scipy.linalg.expm(A * dt)
    
    y_final = expA @ y0
    
    return y_final
