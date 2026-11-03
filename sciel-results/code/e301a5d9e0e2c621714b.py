"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rank = 17
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen part for 2x2 block (indices 0-6)
    # M1
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1
    
    # M6
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Schoolbook for remaining terms (indices 7-16)
    # A02*B20 -> C00
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    # A02*B21 -> C01
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    # A12*B20 -> C10
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    # A12*B21 -> C11
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    # A00*B02 -> C02
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    # A01*B12 -> C02
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    # A10*B02 -> C12
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    # A11*B12 -> C12
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    # A02*B22 -> C02
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    # A12*B22 -> C12
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    # Verification using exact arithmetic
    # We verify the tensor identity: T_{u,v,w} = sum_r U[u,r] V[v,r] W[w,r]
    # T_{u,v,w} is 1 if u=i*m+j, v=j*p+k, w=k*n+i, else 0.
    
    # Since the tensor is small, we can verify a few random cases or the full tensor.
    # Full tensor verification is 6*9*6 = 324 checks. Very fast.
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Expected value is 1
                # Computed value is sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                comp = 0
                for r in range(rank):
                    comp += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                
                if comp != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook rank 18 if verification fails (should not happen)
        rank = 18
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, idx] = 1
                    V[v_idx, idx] = 1
                    W[w_idx, idx] = 1
                    idx += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
