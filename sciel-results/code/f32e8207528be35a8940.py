import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    A_s = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]
    
    B_s = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    
    C_s = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    for block in range(8):
        if block < 4:
            t = block
            is_P = True
        else:
            t = block - 4
            is_P = False
            
        c_idx_base = t * 14
        if is_P:
            c_idx_offset = 0
            a_j_offset = 0
            b_j_offset = 0
        else:
            c_idx_offset = 7
            a_j_offset = 2
            b_j_offset = 2
            
        k_global_offset = t * 2
        
        for s in range(7):
            c_idx = c_idx_base + c_idx_offset + s
            
            for i, j_loc, coeff in A_s[s]:
                j_global = a_j_offset + j_loc
                U[i * m + j_global][c_idx] += coeff
                
            for j_loc, k_loc, coeff in B_s[s]:
                j_global = b_j_offset + j_loc
                k_global = k_global_offset + k_loc
                V[j_global * p + k_global][c_idx] += coeff
                
            for i, k_loc, coeff in C_s[s]:
                k_global = k_global_offset + k_loc
                W[k_global * n + i][c_idx] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
