"""Matrix multiplication 2x4x4 with Strassen-based decomposition (rank 28)."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28  # 4 blocks * 7 Strassen terms
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    # Strassen 2x2x2 coefficients
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For each 2x2x2 block, define which global indices map to local a,b,c,d (A matrix)
    # A is n x m = 2 x 4. We split into 2 blocks along m: A_L = A[:, 0:2], A_R = A[:, 2:4]
    # A_L: indices (i, j) with j in {0,1} -> U indices: i*m + j = i*4 + j
    #   local a = A[0,0] -> U[0], b = A[0,1] -> U[1], c = A[1,0] -> U[4], d = A[1,1] -> U[5]
    # A_R: indices (i, j) with j in {2,3} -> U indices: i*4 + j
    #   local a = A[0,2] -> U[2], b = A[0,3] -> U[3], c = A[1,2] -> U[6], d = A[1,3] -> U[7]
    
    # B is m x p = 4 x 4. Split into 4 blocks:
    # B_TL: rows 0-1, cols 0-1 -> V indices: j*4 + k, j in {0,1}, k in {0,1} -> V[0],V[1],V[4],V[5]
    # B_TR: rows 0-1, cols 2-3 -> V indices: V[2],V[3],V[6],V[7]
    # B_BL: rows 2-3, cols 0-1 -> V indices: V[8],V[9],V[12],V[13]
    # B_BR: rows 2-3, cols 2-3 -> V indices: V[10],V[11],V[14],V[15]
    
    # C is p x n = 4 x 2. W index: k*n + i = k*2 + i
    # C_LL: k in {0,1}, i in {0,1} -> W[0],W[1],W[2],W[3]
    # C_LR: k in {2,3}, i in {0,1} -> W[4],W[5],W[6],W[7]
    
    # Blocks:
    # Block 0: A_L, B_TL, C_LL
    # Block 1: A_L, B_TR, C_LR
    # Block 2: A_R, B_BL, C_LL
    # Block 3: A_R, B_BR, C_LR
    
    # Local indices in each 2x2 matrix (0-indexed):
    # A: a=0, b=1, c=2, d=3
    # B: e=0, f=1, g=2, h=3
    # C: C00=0, C01=1, C10=2, C11=3
    
    # Strassen terms (U_coefs, V_coefs, W_coefs) as (local_idx, coeff) lists
    strassen_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # Block definitions: (A_local_to_global, B_local_to_global, C_local_to_global)
    # A: local 0->a, 1->b, 2->c, 3->d
    # B: local 0->e, 1->f, 2->g, 3->h
    # C: local 0->C00, 1->C01, 2->C10, 3->C11
    
    blocks = [
        # Block 0: A_L, B_TL, C_LL
        ([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 2, 3]),
        # Block 1: A_L, B_TR, C_LR
        ([0, 1, 4, 5], [2, 3, 6, 7], [4, 5, 6, 7]),
        # Block 2: A_R, B_BL, C_LL
        ([2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 2, 3]),
        # Block 3: A_R, B_BR, C_LR
        ([2, 3, 6, 7], [10, 11, 14, 15], [4, 5, 6, 7]),
    ]
    
    for b_idx, (a_map, b_map, c_map) in enumerate(blocks):
        for s_idx, (u_coefs, v_coefs, w_coefs) in enumerate(strassen_terms):
            r = b_idx * 7 + s_idx
            for local_idx, coeff in u_coefs:
                global_idx = a_map[local_idx]
                U[global_idx][r] += coeff
            for local_idx, coeff in v_coefs:
                global_idx = b_map[local_idx]
                V[global_idx][r] += coeff
            for local_idx, coeff in w_coefs:
                global_idx = c_map[local_idx]
                W[global_idx][r] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
