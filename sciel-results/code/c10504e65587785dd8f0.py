import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 42 decomposition using Strassen on 2x2 blocks
    # Structure: 
    # A (2x4) split into A1 (2x2), A2 (2x2)
    # B (4x6) split into B1 (2x6), B2 (2x6)
    # C = A1 B1 + A2 B2
    # Each product (2x2 * 2x6) is decomposed into 3 (2x2 * 2x2) Strassen products.
    # Total 6 Strassen products of rank 7 = 42.
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2x2
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 - X11)(Y10 + Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Local indices for 2x2 matrix:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # Term definitions: (U_indices, V_indices, W_indices)
    # Each index is (local_idx, coeff)
    strassen_terms = [
        {
            'U': [(0, 1), (3, 1)], 
            'V': [(0, 1), (3, 1)], 
            'W': [(0, 1), (3, 1)]
        },
        {
            'U': [(2, 1), (3, 1)], 
            'V': [(0, 1)], 
            'W': [(2, 1), (3, -1)]
        },
        {
            'U': [(0, 1)], 
            'V': [(1, 1), (3, -1)], 
            'W': [(1, 1), (3, 1)]
        },
        {
            'U': [(3, 1)], 
            'V': [(2, 1), (0, -1)], 
            'W': [(2, 1), (0, 1)]
        },
        {
            'U': [(0, 1), (1, 1)], 
            'V': [(3, 1)], 
            'W': [(0, -1), (1, 1)]
        },
        {
            'U': [(2, 1), (0, -1)], 
            'V': [(0, 1), (1, 1)], 
            'W': [(3, 1)]
        },
        {
            'U': [(1, 1), (3, -1)], 
            'V': [(2, 1), (3, 1)], 
            'W': [(0, 1)]
        }
    ]
    
    col = 0
    # r: block of p (3 blocks of 2)
    # s: block of m (2 blocks of 2)
    for r in range(3):
        for s in range(2):
            for t in range(7):
                term = strassen_terms[t]
                
                # U indices
                for local_u, coeff in term['U']:
                    i = local_u // 2
                    j_local = local_u % 2
                    j_global = 2 * s + j_local
                    u_idx = i * m + j_global
                    U[u_idx, col] += coeff
                
                # V indices
                for local_v, coeff in term['V']:
                    j_local = local_v // 2
                    k_local = local_v % 2
                    j_global = 2 * s + j_local
                    k_global = 2 * r + k_local
                    v_idx = j_global * p + k_global
                    V[v_idx, col] += coeff
                
                # W indices
                for local_w, coeff in term['W']:
                    i = local_w // 2
                    k_local = local_w % 2
                    k_global = 2 * r + k_local
                    w_idx = k_global * n + i
                    W[w_idx, col] += coeff
                
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
