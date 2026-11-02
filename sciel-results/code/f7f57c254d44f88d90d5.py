"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    def verify_decomposition(U_arr, V_arr, W_arr, r):
        """Verify exact decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    # Generate schoolbook as fallback
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 140
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # Try to find lower rank using random search
    # The space is very high-dimensional, so random search is unlikely to find exact solutions
    # But let's try a few ranks
    
    for target_rank in [25, 30, 35, 40]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        for trial in range(20):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            if verify_decomposition(U, V, W, target_rank):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
