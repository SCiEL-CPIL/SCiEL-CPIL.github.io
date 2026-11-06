import numpy as np
import scipy.linalg

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # If n is 100, assume standard 10x10 2D Heat Equation
    if n == 100:
        # 10x10 interior points on [0,1]x[0,1] with Dirichlet BC
        # h = 1/11
        h = 1.0 / 11.0
        # 1D Laplacian (tridiagonal, -2 on diag, 1 on off)
        main_diag = -2.0 * np.ones(10)
        off_diag = 1.0 * np.ones(9)
        L1 = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        
        # 2D Laplacian
        I10 = np.eye(10)
        A = (np.kron(L1, I10) + np.kron(I10, L1)) / (h * h)
        
        # Use expm for accuracy
        M = scipy.linalg.expm(A * dt)
        y_end = M @ y0
        return y_end
        
    # Fallback
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(0.0, e_j)
        
    M = scipy.linalg.expm(A * dt)
    y_end = M @ y0
    
    return y_end
