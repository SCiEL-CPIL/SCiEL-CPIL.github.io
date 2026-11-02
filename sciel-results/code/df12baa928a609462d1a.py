"""Optimized program for 2x2x12 matrix multiplication using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin from 150s limit
    
    n, m, p = DIMENSIONS
    rank = 42
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen's 7 linear forms for A (2x2)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_forms = np.array([
        [1, 0, 0, 1],  # S0: a00 + a11
        [0, 0, 1, 1],  # S1: a10 + a11
        [1, 0, 0, 0],  # S2: a00
        [0, 0, 0, 1],  # S3: a11
        [1, 1, 0, 0],  # S4: a00 + a01
        [-1, 0, 1, 0], # S5: a10 - a00
        [0, 1, 0, -1]  # S6: a01 - a11
    ])
    
    # V forms for each block u (0..5)
    # B block u is B[0, 2u:2u+2], B[1, 2u:2u+2]
    # V indices: 0*p + col, 1*p + col
    # col 0 is 2u, col 1 is 2u+1
    
    # W forms
    # Output C[i, k] is at W[k*n + i]
    
    for u in range(6):
        for t in range(7):
            c = u * 7 + t
            
            # Fill U
            U[:, c] = U_forms[t]
            
            # Fill V
            # T0: b00 + b11
            if t == 0:
                V[0 * p + 2*u][c] += 1
                V[1 * p + 2*u + 1][c] += 1
            # T1: b00
            elif t == 1:
                V[0 * p + 2*u][c] += 1
            # T2: b01 - b11
            elif t == 2:
                V[0 * p + 2*u + 1][c] += 1
                V[1 * p + 2*u + 1][c] -= 1
            # T3: b10 - b00
            elif t == 3:
                V[1 * p + 2*u][c] += 1
                V[0 * p + 2*u][c] -= 1
            # T4: b11
            elif t == 4:
                V[1 * p + 2*u + 1][c] += 1
            # T5: b00 + b01
            elif t == 5:
                V[0 * p + 2*u][c] += 1
                V[0 * p + 2*u + 1][c] += 1
            # T6: b10 + b11
            elif t == 6:
                V[1 * p + 2*u][c] += 1
                V[1 * p + 2*u + 1][c] += 1
            
            # Fill W
            # C_00 (i=0, k=2u): M0 + M3 - M4 + M6
            # C_01 (i=0, k=2u+1): M2 + M4
            # C_10 (i=1, k=2u): M1 + M3
            # C_11 (i=1, k=2u+1): M0 - M1 + M2 + M5
            
            if t == 0: # M0
                W[(2*u) * n + 0][c] += 1
                W[(2*u+1) * n + 1][c] += 1
            elif t == 1: # M1
                W[(2*u) * n + 1][c] += 1
                W[(2*u+1) * n + 1][c] -= 1
            elif t == 2: # M2
                W[(2*u+1) * n + 0][c] += 1
                W[(2*u+1) * n + 1][c] += 1
            elif t == 3: # M3
                W[(2*u) * n + 0][c] += 1
                W[(2*u) * n + 1][c] += 1
            elif t == 4: # M4
                W[(2*u) * n + 0][c] -= 1
                W[(2*u+1) * n + 0][c] += 1
            elif t == 5: # M5
                W[(2*u+1) * n + 1][c] += 1
            elif t == 6: # M6
                W[(2*u) * n + 0][c] += 1

    if time.time() > deadline:
         # Fallback if somehow slow (unlikely)
         pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
