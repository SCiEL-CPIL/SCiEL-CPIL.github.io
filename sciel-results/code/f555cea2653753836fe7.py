"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank is 34
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(U_vals, V_vals, W_vals, i_off, j_off, k_off, w_i_off, w_k_off):
        nonlocal col
        if col >= rank:
            return
        # U_vals is list of n*m, V_vals is list of m*p, W_vals is list of p*n
        # We are adding a term for a sub-block
        for i in range(len(U_vals)):
            U[i_off + i][col] += U_vals[i]
        for j in range(len(V_vals)):
            V[j_off + j][col] += V_vals[j]
        for k in range(len(W_vals)):
            W[k_off + k][col] += W_vals[k]
        col += 1

    # Strassen 2x2x2 decomposition
    # U, V, W are 4x1 vectors (flattened 2x2)
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 1]), # C11 += M1, C22 += M1
        # M2
        ([0, 0, 1, 1], [1, 0, 1, 0], [0, 0, 1, 0]), # C21 += M2, C22 -= M2
        # M3
        ([1, 1, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # C12 += M3, C22 += M3
        # M4
        ([0, 0, 0, 1], [1, 0, -1, 0], [1, 0, 0, 1]), # C11 += M4, C21 += M4, C22 -= M4? 
        # Wait, C22 = M1 - M2 + M3 + M6.
        # My W for M4 above was [1, 0, 0, 1] (C11 and C22).
        # But C22 should have -M4? No, C22 = M1 - M2 + M3 + M6.
        # Let's re-verify Strassen formulas.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So:
        # M1: C11 += 1, C22 += 1. W=[1,0,0,1]
        # M2: C21 += 1, C22 -= 1. W=[0,0,1,-1]
        # M3: C12 += 1, C22 += 1. W=[0,1,0,1]
        # M4: C11 += 1, C21 += 1. W=[1,0,1,0]
        # M5: C11 -= 1, C12 += 1. W=[-1,1,0,0]
        # M6: C22 += 1. W=[0,0,0,1]
        # M7: C11 += 1. W=[1,0,0,0]
    ]
    
    # Corrected Strassen terms
    # U: A matrix
    # V: B matrix
    # W: C matrix
    s_terms = [
        # M1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)B00
        ([0, 0, 1, 1], [1, 0, 1, 0], [0, 0, 1, -1]),
        # M3 = A00(B01-B11)
        ([1, 1, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4 = A11(B10-B00)
        ([0, 0, 0, 1], [1, 0, -1, 0], [1, 0, 1, 0]),
        # M5 = (A00+A01)B11
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (A10-A00)(B00+B01)
        ([-1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1]),
        # M7 = (A01-A11)(B10+B11)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # Block 1: A11 (2x2) * B_top (2x4)
    # B_top is rows 0,1 of B.
    # We split B_top into two 2x2 blocks: B_t0 (cols 0,1) and B_t1 (cols 2,3).
    # For B_t0:
    for U_s, V_s, W_s in s_terms:
        # U is 2x2, V is 2x2, W is 2x2
        # V corresponds to B_t0 (cols 0,1)
        V_full = [0]*8
        V_full[0:4] = V_s
        # W corresponds to C_top (rows 0,1) cols 0,1
        W_full = [0]*8
        W_full[0:4] = W_s
        # U corresponds to A11 (rows 0,1 cols 0,1)
        U_full = [0]*9
        U_full[0:4] = U_s
        add_term(U_full, V_full, W_full, 0, 0, 0, 0, 0)
        
    # For B_t1:
    for U_s, V_s, W_s in s_terms:
        V_full = [0]*8
        V_full[4:8] = V_s
        W_full = [0]*8
        W_full[4:8] = W_s
        U_full = [0]*9
        U_full[0:4] = U_s
        add_term(U_full, V_full, W_full, 0, 0, 0, 0, 0)

    # Block 2: A12 (2x1) * B_bot (1x4)
    # A12 is rows 0,1 col 2.
    # B_bot is row 2.
    # Product is 2x4.
    # This is rank 8.
    # A12 = [a0, a1]^T. B_bot = [b0, b1, b2, b3].
    # C = [a0*b0, a0*b1, a0*b2, a0*b3; a1*b0, a1*b1, a1*b2, a1*b3]
    for i in range(2):
        for k in range(4):
            U_full = [0]*9
            U_full[i*3 + 2] = 1 # A[i, 2]
            V_full = [0]*8
            V_full[2*4 + k] = 1 # B[2, k]
            W_full = [0]*8
            W_full[k*3 + i] = 1 # C[i, k]
            add_term(U_full, V_full, W_full, 0, 0, 0, 0, 0)

    # Block 3: A21 (1x2) * B_top (2x4)
    # A21 is row 2, cols 0,1.
    # B_top is rows 0,1.
    # Product is 1x4.
    # Rank 8.
    for j in range(2):
        for k in range(4):
            U_full = [0]*9
            U_full[2*3 + j] = 1 # A[2, j]
            V_full = [0]*8
            V_full[j*4 + k] = 1 # B[j, k]
            W_full = [0]*8
            W_full[k*3 + 2] = 1 # C[2, k]
            add_term(U_full, V_full, W_full, 0, 0, 0, 0, 0)

    # Block 4: A22 (1x1) * B_bot (1x4)
    # A22 is A[2,2].
    # B_bot is B[2,k].
    # Product is 1x4.
    # Rank 4.
    for k in range(4):
        U_full = [0]*9
        U_full[2*3 + 2] = 1
        V_full = [0]*8
        V_full[2*4 + k] = 1
        W_full = [0]*8
        W_full[k*3 + 2] = 1
        add_term(U_full, V_full, W_full, 0, 0, 0, 0, 0)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
