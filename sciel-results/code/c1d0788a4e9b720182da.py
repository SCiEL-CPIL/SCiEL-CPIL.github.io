import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B01
    # M3 = A00B10
    # M4 = (A00 - A11)(B10 + B11)
    # M5 = (A01 + A10)(B00 + B01)
    # M6 = (A01 - A00)(B10 + B11)
    # M7 = (A10 - A11)(B00 - B01)
    
    # Coefficients for A (u), B (v), C (w)
    # u: index i*m + j
    # v: index j*p + k (absolute)
    # w: index k*n + i (absolute)
    
    strassen_terms = [
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,1): 1},
            {(0,1): 1, (1,1): -1}
        ),
        (
            {(0,0): 1},
            {(1,0): 1},
            {(1,0): 1, (1,1): 1}
        ),
        (
            {(0,0): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        (
            {(0,1): 1, (1,0): 1},
            {(0,0): 1, (0,1): 1},
            {(0,0): -1}
        ),
        (
            {(0,1): 1, (0,0): -1},
            {(1,0): 1, (1,1): 1},
            {(1,0): 1}
        ),
        (
            {(1,0): 1, (1,1): -1},
            {(0,0): 1, (0,1): -1},
            {(0,0): 1}
        )
    ]
    
    r = 0
    num_blocks = p // 2 # 8 blocks
    for g in range(num_blocks):
        k_start = 2 * g
        for term in strassen_terms:
            u_dict, v_dict, w_dict = term
            
            for (i, j), val in u_dict.items():
                u_idx = i * m + j
                U[u_idx, r] = val
                
            for (j, k_rel), val in v_dict.items():
                k_abs = k_start + k_rel
                v_idx = j * p + k_abs
                V[v_idx, r] = val
                
            for (i, k_rel), val in w_dict.items():
                k_abs = k_start + k_rel
                w_idx = k_abs * n + i
                W[w_idx, r] = val
                
            r += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
