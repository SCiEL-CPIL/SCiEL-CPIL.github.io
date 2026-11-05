"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Rank minimal decomposition for 1 x m x p is min(m, p)
    # For 2 x m x p, we can split into two 1 x m x p problems if we decouple i
    # Rank = 2 * min(m, p)
    rank = 2 * min(m, p)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 0: i=0
    # r in 0..min(m,p)-1
    for r in range(min(m, p)):
        for j in range(m):
            if j == r:
                U[0 * m + j][r] = 1
                V[j * p + 0][r] = 1 # Wait, V index is (j,k). W index is (k,i).
                # In the sum for i=0:
                # U[0,j] * V[j,k] * W[k,0]
                # We set U[0,j]=1 if j=r.
                # We set W[k,0]=1 for all k.
                # We need V[j,k] such that sum_r U[0,j]_r V[j,k]_r W[k,0]_r = 1.
                # If U[0,j]_r = delta_{j,r} and W[k,0]_r = 1,
                # then sum is V[j,k]_j.
                # We need V[j,k]_j = 1 for all j,k.
                # So V[j,k]_r = 1 if r=j.
                pass
        
        # Correct setup for block i=0 (r in 0..m-1)
        for j in range(m):
            U[0 * m + j][r] = 1 if j == r else 0
            for k in range(p):
                V[j * p + k][r] = 1 if j == r else 0
            for k in range(p):
                W[k * n + 0][r] = 1

    # Block 1: i=1
    # r in m..m+min(m,p)-1
    for r in range(min(m, p)):
        r_global = m + r
        for j in range(m):
            U[1 * m + j][r_global] = 1 if j == r else 0
            for k in range(p):
                V[j * p + k][r_global] = 1 if j == r else 0
            for k in range(p):
                W[k * n + 1][r_global] = 1

    # Verify
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    if verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col] = 1
                    V_s[j*p+k][col] = 1
                    W_s[k*n+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
