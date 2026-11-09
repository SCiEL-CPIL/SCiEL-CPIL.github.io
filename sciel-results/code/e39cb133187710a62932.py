"""Rank-189 decomposition for 6x6 matrix multiplication using 3x3 block Strassen."""
import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # all 6
    
    # Rank 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen coefficients for the inner blocks
    # A: {(0,0):1, (1,1):1}, etc.
    strassen_A = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    strassen_B = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    # W coefficients: list of list of (local_coord, coeff)
    strassen_W = [
        [((0,0), 1), ((1,1), 1)],
        [((1,0), 1), ((1,1), -1)],
        [((0,1), 1), ((1,1), 1)],
        [((0,0), 1), ((1,0), 1)],
        [((0,0), -1), ((0,1), 1)],
        [((1,1), 1)],
        [((0,0), 1)],
    ]
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                if time.time() > deadline:
                    # Return best effort (partial fill is invalid, but we must return something)
                    # Since we fill sequentially, a partial fill is an invalid tensor decomposition.
                    # However, 189 terms are very fast to generate. This should not trigger.
                    pass
                    
                # 2x2 Strassen for each 2x2 block
                # A block at (2*bi, 2*bj)
                # B block at (2*bj, 2*bk)
                # C block at (2*bi, 2*bk)
                
                for t in range(7):
                    # Fill U
                    for (ai, ak), coeff in strassen_A[t].items():
                        global_i = 2*bi + ai
                        global_k = 2*bj + ak
                        u_idx = global_i * m + global_k
                        U[u_idx][col] = coeff
                        
                    # Fill V
                    for (bk_local, bj_local), coeff in strassen_B[t].items():
                        global_bk = 2*bj + bk_local
                        global_bj = 2*bk + bj_local
                        v_idx = global_bk * p + global_bj
                        V[v_idx][col] = coeff
                        
                    # Fill W
                    # Trace convention: W[k*n + i, r]
                    # Here local C indices are (ci, cj) where ci is row, cj is col of C block
                    # Global C row = 2*bi + ci
                    # Global C col = 2*bk + cj
                    # W index = (Global C col) * n + (Global C row)
                    for (ci, cj), coeff in strassen_W[t]:
                        global_i = 2*bi + ci
                        global_j = 2*bk + cj
                        w_idx = global_j * n + global_i
                        W[w_idx][col] = coeff
                        
                    col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
