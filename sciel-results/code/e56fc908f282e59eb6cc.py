"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 150.0  # 150s wall-clock limit, use 145s for safety
    
    n, m, p = DIMENSIONS
    
    # Rank 11 decomposition for M_{2,2,3}
    # We split the 9 columns into 3 blocks of 3.
    # Each block uses Strassen (rank 7) for first 2 cols and Schoolbook (rank 4) for 3rd col.
    
    total_rank = 33
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for block in range(3):
        k0 = block * 3
        k1 = block * 3 + 1
        k2 = block * 3 + 2
        
        # Strassen part for columns k0, k1
        # U factors (7)
        strassen_U = [
            [1, 0, 0, 1],   # M1
            [0, 0, 1, 1],   # M2
            [1, 0, 0, 0],   # M3
            [0, 0, 0, 1],   # M4
            [1, 1, 0, 0],   # M5
            [-1, 0, 1, 0],  # M6
            [0, 1, 0, -1]   # M7
        ]
        
        # V factors (7) - indices in B (size 18)
        # B indices: j*p + k. j=0,1.
        # k0, k1 are the column indices in B.
        strassen_V = [
            {k0: 1, 9+k1: 1},          # M1: e0 + f1
            {k0: 1},                   # M2: e0
            {9+k0: 1, 9+k1: -1},       # M3: f0 - f1
            {k1: 1, k0: -1},           # M4: e1 - e0
            {9+k1: 1},                 # M5: f1
            {k0: 1, 9+k0: 1},          # M6: e0 + f0
            {k1: 1, 9+k1: 1}           # M7: e1 + f1
        ]
        
        # W factors (7) - indices in C (size 18)
        # C indices: k_out * n + i. k_out=k0, k1. i=0,1.
        strassen_W = [
            {k0*2+0: 1, k1*2+1: 1},    # M1
            {k0*2+1: 1},               # M2
            {k1*2+0: 1, k1*2+1: -1},   # M3
            {k0*2+0: -1, k1*2+0: 1},   # M4
            {k0*2+0: 1},               # M5
            {k0*2+1: 1, k1*2+1: 1},    # M6
            {k0*2+0: 1, k1*2+1: 1}     # M7
        ]
        
        for c in range(7):
            for i in range(n*m):
                U[i][col] = strassen_U[c][i]
            for idx, val in strassen_V[c].items():
                V[idx][col] = val
            for idx, val in strassen_W[c].items():
                W[idx][col] = val
            col += 1
        
        # Schoolbook part for column k2
        # 4 terms
        schoolbook_U = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        schoolbook_V = [
            {k2: 1},
            {9+k2: 1},
            {k2: 1},
            {9+k2: 1}
        ]
        schoolbook_W = [
            {k2*2+0: 1},
            {k2*2+0: 1},
            {k2*2+1: 1},
            {k2*2+1: 1}
        ]
        
        for c in range(4):
            for i in range(n*m):
                U[i][col] = schoolbook_U[c][i]
            for idx, val in schoolbook_V[c].items():
                V[idx][col] = val
            for idx, val in schoolbook_W[c].items():
                W[idx][col] = val
            col += 1
            
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-10:
                        return False
        return True

    if verify(result["U"], result["V"], result["W"]):
        return result
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
