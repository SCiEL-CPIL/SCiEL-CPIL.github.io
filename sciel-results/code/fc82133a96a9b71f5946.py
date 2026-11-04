"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # Part 1: j=0 (2x1x8), rank 16
    # Part 2: j=1,2 (2x2x8), split into 4 blocks of 2x2x2, rank 4*7=28
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: j=0
    col = 0
    for i in range(n):
        for k in range(p):
            # U index: i*m + j = i*3 + 0
            U[i * m][col] = 1
            # V index: j*p + k = 0*8 + k = k
            V[k][col] = 1
            # W index: k*n + i = k*2 + i
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j=1,2
    # Strassen for 2x2x2
    # A_sub = [[A01, A02], [A11, A12]]
    # B_sub = [[B1k, B1k+1], [B2k, B2k+1]]
    
    # U coeffs for A_sub
    # L1 = A01
    # L2 = A01 + A02
    # L3 = A11 + A12
    # L4 = A12
    # L5 = A01 + A11
    # L6 = A02 - A12
    # L7 = A01 - A11
    
    # Indices in U:
    # A01 -> 3*0 + 1 = 1
    # A02 -> 3*0 + 2 = 2
    # A11 -> 3*1 + 1 = 4
    # A12 -> 3*1 + 2 = 5
    
    U_coeffs = [
        {1: 1},                                  # L1
        {1: 1, 2: 1},                            # L2
        {4: 1, 5: 1},                            # L3
        {5: 1},                                  # L4
        {1: 1, 4: 1},                            # L5
        {2: 1, 5: -1},                           # L6
        {1: 1, 4: -1}                            # L7
    ]
    
    # V coeffs for B_sub (relative to block t)
    # B1k -> 8*1 + k = 8+k
    # B1k+1 -> 8*1 + k+1 = 9+k
    # B2k -> 8*2 + k = 16+k
    # B2k+1 -> 8*2 + k+1 = 17+k
    
    # R1 = B1k + B2k+1
    # R2 = B2k+1
    # R3 = B1k
    # R4 = B2k - B2k+1
    # R5 = B1k + B1k+1
    # R6 = B2k + B2k+1
    # R7 = B1k+1 + B2k+1
    
    # W coeffs for C_sub
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M5
    # C10 = M3 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # W index: k*n + i
    # For block t, k = 2t + k'
    # C00 (i=0, k'=0) -> 2(2t) + 0 = 4t
    # C01 (i=0, k'=1) -> 2(2t+1) + 0 = 4t+2
    # C10 (i=1, k'=0) -> 2(2t) + 1 = 4t+1
    # C11 (i=1, k'=1) -> 2(2t+1) + 1 = 4t+3
    
    # W_coeffs[s] = {w_index_offset: coeff}
    # w_index_offset is relative to 4t
    W_coeffs = [
        {0: 1, 3: 1},    # M1 -> C00, C11
        {2: 1, 3: -1},   # M2 -> C01, -C11
        {1: 1, 3: 1},    # M3 -> C10, C11
        {0: 1, 3: 1},    # M4 -> C00, C11
        {0: -1, 2: 1},   # M5 -> -C00, C01
        {1: 1},          # M6 -> C10
        {0: 1}           # M7 -> C00
    ]
    
    for t in range(4):
        base_k = 2 * t
        for s in range(7):
            col = 16 + t * 7 + s
            
            # Update U
            for idx, val in U_coeffs[s].items():
                U[idx][col] = val
                
            # Update V
            # R1
            if s == 0:
                V[8 + base_k][col] = 1
                V[16 + base_k + 1][col] = 1
            # R2
            elif s == 1:
                V[16 + base_k + 1][col] = 1
            # R3
            elif s == 2:
                V[8 + base_k][col] = 1
            # R4
            elif s == 3:
                V[16 + base_k][col] = 1
                V[16 + base_k + 1][col] = -1
            # R5
            elif s == 4:
                V[8 + base_k][col] = 1
                V[8 + base_k + 1][col] = 1
            # R6
            elif s == 5:
                V[16 + base_k][col] = 1
                V[16 + base_k + 1][col] = 1
            # R7
            elif s == 6:
                V[8 + base_k + 1][col] = 1
                V[16 + base_k + 1][col] = 1
                
            # Update W
            base_w = 4 * t
            for idx_offset, val in W_coeffs[s].items():
                W[base_w + idx_offset][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
