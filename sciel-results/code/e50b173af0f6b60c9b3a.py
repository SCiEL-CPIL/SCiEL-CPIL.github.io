import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank calculation: 6 blocks of 2x2x2 (rank 7) + 1 block of 2x2x1 (rank 4)
    num_blocks_2x2 = p // 2
    num_blocks_2x1 = p % 2
    rank = num_blocks_2x2 * 7 + num_blocks_2x1 * 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen terms for 2x2x2
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 1, 0, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Process 2x2 blocks
    for b in range(num_blocks_2x2):
        # Column indices for this block
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Global V indices for B
        # B[0, k0] -> 0*p + k0
        # B[0, k1] -> 0*p + k1
        # B[1, k0] -> 1*p + k0
        # B[1, k1] -> 1*p + k1
        v_idx_00 = 0 * p + k0
        v_idx_01 = 0 * p + k1
        v_idx_10 = 1 * p + k0
        v_idx_11 = 1 * p + k1
        
        # Global W indices for C
        # C[0, k0] -> k0*n + 0
        # C[0, k1] -> k1*n + 0
        # C[1, k0] -> k0*n + 1
        # C[1, k1] -> k1*n + 1
        w_idx_00 = k0 * n + 0
        w_idx_01 = k1 * n + 0
        w_idx_10 = k0 * n + 1
        w_idx_11 = k1 * n + 1
        
        # Local V indices mapping: 0->00, 1->01, 2->10, 3->11
        # Local W indices mapping: 0->00, 1->01, 2->10, 3->11
        
        for u_c, v_c, w_c in strassen_terms:
            # U
            for i in range(4):
                if u_c[i] != 0:
                    U[i][col] = u_c[i]
            
            # V
            # Map local v indices to global
            if v_c[0] != 0: V[v_idx_00][col] = v_c[0]
            if v_c[1] != 0: V[v_idx_01][col] = v_c[1]
            if v_c[2] != 0: V[v_idx_10][col] = v_c[2]
            if v_c[3] != 0: V[v_idx_11][col] = v_c[3]
            
            # W
            if w_c[0] != 0: W[w_idx_00][col] = w_c[0]
            if w_c[1] != 0: W[w_idx_01][col] = w_c[1]
            if w_c[2] != 0: W[w_idx_10][col] = w_c[2]
            if w_c[3] != 0: W[w_idx_11][col] = w_c[3]
            
            col += 1
            
    # Process 2x1 block
    if num_blocks_2x1 > 0:
        k0 = 2 * num_blocks_2x2
        
        v_idx_00 = 0 * p + k0
        v_idx_10 = 1 * p + k0
        
        w_idx_00 = k0 * n + 0
        w_idx_10 = k0 * n + 1
        
        # Terms:
        # 1. A00 * B00 -> C00
        U[0][col] = 1
        V[v_idx_00][col] = 1
        W[w_idx_00][col] = 1
        col += 1
        
        # 2. A01 * B10 -> C00
        U[1][col] = 1
        V[v_idx_10][col] = 1
        W[w_idx_00][col] = 1
        col += 1
        
        # 3. A10 * B00 -> C10
        U[2][col] = 1
        V[v_idx_00][col] = 1
        W[w_idx_10][col] = 1
        col += 1
        
        # 4. A11 * B10 -> C10
        U[3][col] = 1
        V[v_idx_10][col] = 1
        W[w_idx_10][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
