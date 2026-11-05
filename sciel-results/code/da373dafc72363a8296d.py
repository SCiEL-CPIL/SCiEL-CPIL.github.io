"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U (A), V (B), W (C)
    # Terms: (U_coeffs, V_coeffs, W_coeffs)
    # Coeffs are lists of (local_idx, coeff)
    
    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]
    
    # Blocks:
    # 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    blocks = [
        # (i_start, j_A_start, j_B_start, k_C_start)
        (0, 0, 0, 0),
        (0, 2, 2, 0),
        (0, 0, 0, 2),
        (0, 2, 2, 2)
    ]
    
    col_offset = 0
    for block in blocks:
        i_start, j_A_start, j_B_start, k_C_start = block
        
        for t_idx, (u_coeffs, v_coeffs, w_coeffs) in enumerate(strassen_terms):
            col = col_offset + t_idx
            
            # Update U
            # U row: i * m + j
            # Local A index: r * 2 + c
            for local_idx, coeff in u_coeffs:
                r, c = divmod(local_idx, 2)
                i_global = i_start + r
                j_global = j_A_start + c
                U[i_global * m + j_global][col] += coeff
                
            # Update V
            # V row: j * p + k
            # Local B index: r * 2 + c
            for local_idx, coeff in v_coeffs:
                r, c = divmod(local_idx, 2)
                j_global = j_B_start + r
                k_global = k_C_start + c
                V[j_global * p + k_global][col] += coeff
                
            # Update W
            # W row: k * n + i
            # Local C index: r * 2 + c
            for local_idx, coeff in w_coeffs:
                r, c = divmod(local_idx, 2)
                i_global = i_start + r
                k_global = k_C_start + c
                W[k_global * n + i_global][col] += coeff
        
        col_offset += 7
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
