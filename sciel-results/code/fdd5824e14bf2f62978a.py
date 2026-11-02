"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based rank-14 decomposition.
    # The problem 2x2 * 2x4 is equivalent to two independent 2x2 * 2x2 multiplications.
    # Rank of 2x2 * 2x2 is 7 (Strassen). Total rank 7 + 7 = 14.
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U indices: i*m + j. i in {0,1}, j in {0,1}.
    # a=U[0], b=U[1], c=U[2], d=U[3]
    
    # V indices: j*p + k. j in {0,1}, k in {0,1,2,3}.
    # Block 1 (k=0,1): e=V[0], f=V[1], g=V[4], h=V[5]
    # Block 2 (k=2,3): e'=V[2], f'=V[3], g'=V[6], h'=V[7]
    
    # W indices: k*n + i. k in {0,1,2,3}, i in {0,1}.
    # Block 1: C00=W[0], C01=W[2], C10=W[1], C11=W[3]
    # Block 2: C02=W[4], C03=W[6], C12=W[5], C13=W[7]

    def set_strassen_block(offset_u, offset_v, offset_w, v_indices, w_indices):
        # offset_u: column offset in U (0 for both blocks as U is shared structure)
        # offset_v: column offset in V (0 for block 1, 7 for block 2)
        # offset_w: column offset in W (0 for block 1, 7 for block 2)
        
        # M1: (a+d)(e+h)
        U[0][offset_u + 0] = 1
        U[3][offset_u + 0] = 1
        V[v_indices[0]][offset_v + 0] = 1
        V[v_indices[3]][offset_v + 0] = 1
        
        # M2: (c+d)e
        U[2][offset_u + 1] = 1
        U[3][offset_u + 1] = 1
        V[v_indices[0]][offset_v + 1] = 1
        
        # M3: a(f-h)
        U[0][offset_u + 2] = 1
        V[v_indices[1]][offset_v + 2] = 1
        V[v_indices[3]][offset_v + 2] = -1
        
        # M4: d(g-e)
        U[3][offset_u + 3] = 1
        V[v_indices[2]][offset_v + 3] = 1
        V[v_indices[0]][offset_v + 3] = -1
        
        # M5: (a+b)h
        U[0][offset_u + 4] = 1
        U[1][offset_u + 4] = 1
        V[v_indices[3]][offset_v + 4] = 1
        
        # M6: (c-a)(e+f)
        U[2][offset_u + 5] = 1
        U[0][offset_u + 5] = -1
        V[v_indices[0]][offset_v + 5] = 1
        V[v_indices[1]][offset_v + 5] = 1
        
        # M7: (b-d)(g+h)
        U[1][offset_u + 6] = 1
        U[3][offset_u + 6] = -1
        V[v_indices[2]][offset_v + 6] = 1
        V[v_indices[3]][offset_v + 6] = 1
        
        # Reconstruction W
        # C00 = M1 + M4 - M5 + M7
        W[w_indices[0]][offset_w + 0] = 1
        W[w_indices[0]][offset_w + 3] = 1
        W[w_indices[0]][offset_w + 4] = -1
        W[w_indices[0]][offset_w + 6] = 1
        
        # C01 = M3 + M5
        W[w_indices[2]][offset_w + 2] = 1
        W[w_indices[2]][offset_w + 4] = 1
        
        # C10 = M2 + M4
        W[w_indices[1]][offset_w + 1] = 1
        W[w_indices[1]][offset_w + 3] = 1
        
        # C11 = M1 - M2 + M3 + M6
        W[w_indices[3]][offset_w + 0] = 1
        W[w_indices[3]][offset_w + 1] = -1
        W[w_indices[3]][offset_w + 2] = 1
        W[w_indices[3]][offset_w + 5] = 1

    # Block 1: k=0,1
    # V indices for B1: e=V[0], f=V[1], g=V[4], h=V[5]
    v_idx_1 = [0, 1, 4, 5]
    # W indices for C1: C00=W[0], C10=W[1], C01=W[2], C11=W[3]
    # Note: w_indices order in function is [C00, C10, C01, C11]
    w_idx_1 = [0, 1, 2, 3]
    set_strassen_block(0, 0, 0, v_idx_1, w_idx_1)

    # Block 2: k=2,3
    # V indices for B2: e'=V[2], f'=V[3], g'=V[6], h'=V[7]
    v_idx_2 = [2, 3, 6, 7]
    # W indices for C2: C02=W[4], C12=W[5], C03=W[6], C13=W[7]
    w_idx_2 = [4, 5, 6, 7]
    # U is the same, so offset_u is 0 again (overwriting same values, which is fine as they are identical)
    # But wait, U columns 0-6 are used for block 1.
    # We need U columns 7-13 to be the same as 0-6.
    # My function call above sets U[...][0..6].
    # I need to set U[...][7..13] to the same values.
    # So I call the function again with offset_u=7, offset_v=7, offset_w=7.
    set_strassen_block(7, 7, 7, v_idx_2, w_idx_2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
