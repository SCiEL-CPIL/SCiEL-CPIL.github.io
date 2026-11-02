import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Build the multiplication tensor T
    T = np.zeros((p*n, n*m, m*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k*n + i
                b = i*m + j
                c = j*p + k
                T[a, b, c] = 1
    
    # Schoolbook decomposition
    rank_sb = n*m*p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to reduce the rank by finding linear dependencies
    # For now, just return the schoolbook decomposition
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
