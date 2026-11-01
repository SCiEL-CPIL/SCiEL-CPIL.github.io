"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 using Strassen decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions
    # X indices in local 2x2: 0:00, 1:01, 2:10, 3:11
    # Y indices in local 2x2: 0:00, 1:01, 2:10, 3:11
    # Terms: (X_coeffs, Y_coeffs, W_contribs)
    # W_contribs: {r00: val, r01: val, r10: val, r11: val}
    
    terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    for t in range(rank):
        block_idx = t // 7
        term_idx = t % 7
        
        x_coeffs, y_coeffs, w_contribs = terms[term_idx]
        
        if block_idx < 4:
            # Top block, A_left
            k = block_idx
            # X mapping (A_left)
            # 0: A[0,0] -> U[0]
            # 1: A[0,1] -> U[1]
            # 2: A[1,0] -> U[4]
            # 3: A[1,1] -> U[5]
            x_map = {0: 0, 1: 1, 2: 4, 3: 5}
            
            # Y mapping (B_top, k)
            # 0: B[0, 2k] -> V[2k]
            # 1: B[0, 2k+1] -> V[2k+1]
            # 2: B[1, 2k] -> V[8+2k]
            # 3: B[1, 2k+1] -> V[8+2k+1]
            y_map = {0: 2*k, 1: 2*k+1, 2: 8+2*k, 3: 8+2*k+1}
            
            # W mapping (C cols 2k, 2k+1)
            # r00: C[0, 2k] -> W[4k]
            # r01: C[0, 2k+1] -> W[4k+1]
            # r10: C[1, 2k] -> W[4k+2]
            # r11: C[1, 2k+1] -> W[4k+3]
            w_map = {0: 4*k, 1: 4*k+1, 2: 4*k+2, 3: 4*k+3}
        else:
            # Bottom block, A_right
            k = block_idx - 4
            # X mapping (A_right)
            # 0: A[0,2] -> U[2]
            # 1: A[0,3] -> U[3]
            # 2: A[1,2] -> U[6]
            # 3: A[1,3] -> U[7]
            x_map = {0: 2, 1: 3, 2: 6, 3: 7}
            
            # Y mapping (B_bot, k)
            # 0: B[2, 2k] -> V[16+2k]
            # 1: B[2, 2k+1] -> V[16+2k+1]
            # 2: B[3, 2k] -> V[24+2k]
            # 3: B[3, 2k+1] -> V[24+2k+1]
            y_map = {0: 16+2*k, 1: 16+2*k+1, 2: 24+2*k, 3: 24+2*k+1}
            
            # W mapping (C cols 2k, 2k+1)
            # r00: C[0, 2k] -> W[4k]
            # r01: C[0, 2k+1] -> W[4k+1]
            # r10: C[1, 2k] -> W[4k+2]
            # r11: C[1, 2k+1] -> W[4k+3]
            w_map = {0: 4*k, 1: 4*k+1, 2: 4*k+2, 3: 4*k+3}
            
        # Fill U
        for x_idx, coeff in x_coeffs.items():
            U[x_map[x_idx]][t] += coeff
            
        # Fill V
        for y_idx, coeff in y_coeffs.items():
            V[y_map[y_idx]][t] += coeff
            
        # Fill W
        for r_idx, coeff in w_contribs.items():
            W[w_map[r_idx]][t] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
