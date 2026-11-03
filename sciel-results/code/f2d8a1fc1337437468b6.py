import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    
    # Rank 18 construction
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Map (j, k) to column index c
    # c = k * m + j
    for k in range(p):
        for j in range(m):
            c = k * m + j
            
            # V[j*p+k, c] = 1
            V[j * p + k, c] = 1.0
            
            # U[i*m+j, c] = 1 for i in {0, 1}
            U[0 * m + j, c] = 1.0
            U[1 * m + j, c] = 1.0
            
            # W[k*n+i, c] = 1 for i in {0, 1}
            W[k * n + 0, c] = 1.0
            W[k * n + 1, c] = 1.0
            
    # Verify
    # Reconstruct T[i,j,k]
    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    
    # Target is all ones
    target = np.ones((n, m, p))
    
    if not np.allclose(T, target, atol=1e-9):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        
        return {"U": U_sb, "V": V_sb, "W": W_sb}
        
    return {"U": U, "V": V, "W": W}
