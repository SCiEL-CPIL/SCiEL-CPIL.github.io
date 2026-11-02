import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 4, 8
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Define the 8 blocks
    # u in 0,1 ; v in 0,1,2,3
    # block_id = 4*u + v
    # base_col = 7 * block_id
    
    # Indices for A (U), B (V), C (W)
    # A: i*m + j
    # B: j*p + k
    # C: k*n + i
    
    # Precompute indices for efficiency? 
    # Or just loop.
    
    for u in range(2):
        for v in range(4):
            base = 7 * (4 * u + v)
            
            # A indices
            a11 = 0 * m + 2 * u
            a12 = 0 * m + 2 * u + 1
            a21 = 1 * m + 2 * u
            a22 = 1 * m + 2 * u + 1
            
            # B indices
            b11 = 2 * u * p + 2 * v
            b12 = 2 * u * p + 2 * v + 1
            b21 = (2 * u + 1) * p + 2 * v
            b22 = (2 * u + 1) * p + 2 * v + 1
            
            # C indices (W convention k*n + i)
            c11 = 2 * v * n + 0
            c12 = (2 * v + 1) * n + 0
            c21 = 2 * v * n + 1
            c22 = (2 * v + 1) * n + 1
            
            # M1
            U[a11, base] += 1; U[a22, base] += 1
            V[b11, base] += 1; V[b22, base] += 1
            W[c11, base] += 1; W[c22, base] += 1
            
            # M2
            U[a21, base+1] += 1; U[a22, base+1] += 1
            V[b11, base+1] += 1
            W[c21, base+1] += 1; W[c22, base+1] -= 1
            
            # M3
            U[a11, base+2] += 1
            V[b12, base+2] += 1; V[b22, base+2] -= 1
            W[c12, base+2] += 1; W[c22, base+2] += 1
            
            # M4
            U[a22, base+3] += 1
            V[b21, base+3] += 1; V[b11, base+3] -= 1
            W[c11, base+3] += 1; W[c21, base+3] += 1
            
            # M5
            U[a11, base+4] += 1; U[a12, base+4] += 1
            V[b22, base+4] += 1
            W[c11, base+4] -= 1; W[c12, base+4] += 1
            
            # M6
            U[a21, base+5] += 1; U[a11, base+5] -= 1
            V[b11, base+5] += 1; V[b12, base+5] += 1
            W[c22, base+5] += 1
            
            # M7
            U[a12, base+6] += 1; U[a22, base+6] -= 1
            V[b21, base+6] += 1; V[b22, base+6] += 1
            W[c11, base+6] += 1

    return {"U": U, "V": V, "W": W}
