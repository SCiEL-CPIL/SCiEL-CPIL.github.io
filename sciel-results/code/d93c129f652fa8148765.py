import numpy as np
import time

DIMENSIONS = [2, 2, 2]

def construct():
    n, m, p = DIMENSIONS
    # Strassen's algorithm parameters
    # U: 4x7, V: 4x7
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, -1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    # We need to find W (4x7) such that 
    # C_flat = W @ ( (U @ A_flat) * (V @ B_flat) )
    # This is equivalent to finding W such that the tensor decomposition holds.
    # T_{i,j,k} = sum_r W[k,r] U[i,r] V[j,r]
    # We can solve for W using random samples.
    
    rank = 7
    # Generate random samples
    np.random.seed(42)
    N_samples = 100
    A_samples = np.random.rand(N_samples, 4)
    B_samples = np.random.rand(N_samples, 4)
    
    # Compute true C_flat (col-major: C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # Flat C: [ae+bg, ce+dg, af+bh, cf+dh]
    
    C_true = np.zeros((N_samples, 4))
    for i in range(N_samples):
        a, b, c, d = A_samples[i]
        e, f, g, h = B_samples[i]
        C_true[i, 0] = a*e + b*g
        C_true[i, 1] = c*e + d*g
        C_true[i, 2] = a*f + b*h
        C_true[i, 3] = c*f + d*h
        
    # Compute M_r = (U @ A)_r * (V @ B)_r
    # M shape: (N_samples, rank)
    UA = A_samples @ U.T  # (N, 7)
    VB = B_samples @ V.T  # (N, 7)
    M = UA * VB           # (N, 7)
    
    # Solve C_true = M @ W.T ?
    # C_true[k] = sum_r M[r] W[k,r]
    # So C_true = M @ W.T
    # W.T = M.T @ C_true (pseudo inverse)
    # W = (M.T @ C_true).T
    
    W_T, residuals, rank_sol, s = np.linalg.lstsq(M.T, C_true, rcond=None)
    W = W_T.T
    
    # Round to integers if close
    W_int = np.round(W)
    if np.allclose(W, W_int, atol=1e-9):
        W = W_int
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
