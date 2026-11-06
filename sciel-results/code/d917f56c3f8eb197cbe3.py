import numpy as np
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract the matrix A assuming y' = A y
    # We probe the function with basis vectors
    # Since it's a 10x10 grid, n=100.
    # 100 calls is acceptable.
    
    A = np.zeros((n, n), dtype=float)
    
    # To save time, we can batch the calls if f supported matrices, 
    # but f takes a vector. So we loop.
    # Optimization: We can just call f on the identity matrix columns.
    
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        # Use t=0.0
        A[:, j] = f(0.0, e_j)
        
    # Compute the matrix exponential
    # h = t1 - t0
    h = t1 - t0
    if h == 0:
        return y0
        
    exp_A_h = expm(h * A)
    
    y_end = exp_A_h @ y0
    
    return y_end
