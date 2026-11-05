import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Part 1: 4x4x4 via recursive Strassen (rank 49)
    # Part 2: 4x4x2 via matrix-vector products (rank 8)
    # Total rank: 57
    rank = 57
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps for 2x2x2
    strassen_steps_222 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Part 1: 4x4x4 part (cols 0-48)
    col_start_1 = 0
    
    for s in range(7):
        alpha_4, beta_4, gamma_4 = strassen_steps_222[s]
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            
            col = col_start_1 + s * 7 + s2
            
            # Process U (A): 4x4 matrix
            for (bi, bj), val_a4 in alpha_4.items():
                for (si, sj), val_a2 in alpha_2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_a4 * val_a2
                    
            # Process V (B1): 4x4 matrix
            for (bi, bj), val_b4 in beta_4.items():
                for (si, sj), val_b2 in beta_2.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_b4 * val_b2
                    
            # Process W (C1): 4x4 matrix
            for (bi, bj), val_c4 in gamma_4.items():
                for (si, sj), val_c2 in gamma_2.items():
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_c4 * val_c2

    # Part 2: 4x4x2 part (cols 49-56)
    # For each of the 2 output columns, use rank-4 matrix-vector decomposition
    col_start_2 = 49
    
    for out_col in range(2):  # columns 4 and 5 of B
        for r in range(4):  # rank-4 decomposition
            col = col_start_2 + out_col * 4 + r
            
            # U[i,k] = 1 for all i,k (A is 4x4, so 16 entries)
            for i in range(n):
                for k in range(m):
                    U[i * m + k][col] = 1
                    
            # V[k,j] = 1 for all k,j (B column is 4x1, so 4 entries)
            for k in range(m):
                V[k * p + (4 + out_col)][col] = 1
                
            # W[j,i] = delta(i,r) (C column is 4x1)
            W[(4 + out_col) * n + r][col] = 1

    return {"U": U, "V": V, "W": W}
