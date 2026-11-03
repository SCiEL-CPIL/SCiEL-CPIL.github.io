import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 48 decomposition
    # 8 blocks of 2x2x2, each using 6 terms (Strassen minus P6)
    rank = 48
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients (excluding P6 which is index 5)
    # Indices: 0(M1), 1(M2), 2(M3), 3(M4), 4(M5), 6(M7)
    # We will just use 6 entries in the lists, mapping to these.
    
    u_coeffs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [0, 1, 0, -1]  # M7 (formerly index 6)
    ]
    
    v_coeffs = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1],  # M5
        [0, 0, 1, 1]   # M7
    ]
    
    w_coeffs = [
        [1, 0, 0, 1],   # M1: z00(1), z11(1)
        [0, 0, 1, -1],  # M2: z10(1), z11(-1)
        [0, 1, 0, 1],   # M3: z01(1), z11(1)
        [1, 0, 1, 0],   # M4: z00(1), z10(1)
        [-1, 1, 0, 0],  # M5: z00(-1), z01(1)
        [0, 0, 0, 0]    # M7 placeholder
    ]
    # M7: z00(1), z11(1) ?
    # C11 = P1 + P4 - P5 + P7
    # C22 = P1 - P2 + P3 + P7
    # So M7 (P7) contributes 1 to z00 and 1 to z11.
    w_coeffs[5] = [1, 0, 0, 1]
    
    num_terms = 6
    
    for a_block in range(2):
        for b_block in range(4):
            for t in range(num_terms):
                # Update U
                for i in range(2):
                    for j_loc in range(2):
                        coeff = u_coeffs[t][i * 2 + j_loc]
                        if coeff != 0:
                            row_u = i * m + (a_block * 2 + j_loc)
                            U[row_u][col] += coeff
                
                # Update V
                b_row_offset = a_block * 2
                for j in range(2):
                    for k_loc in range(2):
                        coeff = v_coeffs[t][j * 2 + k_loc]
                        if coeff != 0:
                            j_global = b_row_offset + j
                            k_global = b_block * 2 + k_loc
                            row_v = j_global * p + k_global
                            V[row_v][col] += coeff
                
                # Update W
                for i in range(2):
                    for k_loc in range(2):
                        coeff = w_coeffs[t][i * 2 + k_loc]
                        if coeff != 0:
                            k_global = b_block * 2 + k_loc
                            row_w = k_global * n + i
                            W[row_w][col] += coeff
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
