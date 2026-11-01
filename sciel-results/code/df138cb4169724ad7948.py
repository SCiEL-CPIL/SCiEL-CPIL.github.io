"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 3x8 matrix B into 4 blocks of 3x2.
    # For each block, we use Strassen for the 2x2 part of A and B, 
    # and schoolbook for the 2x1 part of A and 1x2 part of B.
    # Rank per block: 7 (Strassen) + 4 (Schoolbook) = 11.
    # Total rank: 4 * 11 = 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Indices for A (2x2 part): 0, 1, 2, 3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
    # Indices for B (2x2 part): 0, 1, 2, 3 correspond to B[0,0], B[0,1], B[1,0], B[1,1]
    # Indices for C (2x2 part): 0, 1, 2, 3 correspond to C[0,0], C[0,1], C[1,0], C[1,1]
    
    strassen_U = [
        [1, 0, 1, 0], # M1: A00, A11
        [0, 0, 1, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00, A01
        [1, 0, -1, 0],# M6: A10, -A00
        [0, 1, 0, -1] # M7: A01, -A11
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1: B00, B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01, -B11
        [0, 0, 1, -1],# M4: B10, -B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00, B01
        [0, 0, 1, 1]  # M7: B10, B11
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, -1],# M2: C10, -C11
        [0, 1, 0, 1], # M3: C01, C11
        [1, 0, 1, 0], # M4: C00, C10
        [-1, 1, 0, 0],# M5: -C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    for b in range(4):
        # k indices for this block of B: 2b, 2b+1
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # --- Strassen part (7 terms) ---
        for i in range(7):
            # U contributions
            # A indices: 0->A[0,0], 1->A[0,1], 2->A[1,0], 3->A[1,1]
            # U row index: row * m + col
            if strassen_U[i][0] != 0: U[0 * m + 0][col] += strassen_U[i][0]
            if strassen_U[i][1] != 0: U[0 * m + 1][col] += strassen_U[i][1]
            if strassen_U[i][2] != 0: U[1 * m + 0][col] += strassen_U[i][2]
            if strassen_U[i][3] != 0: U[1 * m + 1][col] += strassen_U[i][3]
            
            # V contributions
            # B indices: 0->B[0,k0], 1->B[0,k1], 2->B[1,k0], 3->B[1,k1]
            # V row index: row * p + col
            if strassen_V[i][0] != 0: V[0 * p + k0][col] += strassen_V[i][0]
            if strassen_V[i][1] != 0: V[0 * p + k1][col] += strassen_V[i][1]
            if strassen_V[i][2] != 0: V[1 * p + k0][col] += strassen_V[i][2]
            if strassen_V[i][3] != 0: V[1 * p + k1][col] += strassen_V[i][3]
            
            # W contributions
            # C indices: 0->C[0,k0], 1->C[0,k1], 2->C[1,k0], 3->C[1,k1]
            # W row index: row * n + col (Note: W is p x n, index k*n+i)
            if strassen_W[i][0] != 0: W[k0 * n + 0][col] += strassen_W[i][0]
            if strassen_W[i][1] != 0: W[k1 * n + 0][col] += strassen_W[i][1]
            if strassen_W[i][2] != 0: W[k0 * n + 1][col] += strassen_W[i][2]
            if strassen_W[i][3] != 0: W[k1 * n + 1][col] += strassen_W[i][3]
            
            col += 1
            
        # --- Schoolbook part for A_col2 (2x1) * B_row2 (1x2) (4 terms) ---
        # A index: 4 (A[0,2]), 5 (A[1,2])
        # B index: 6+k0 (B[2,k0]), 6+k1 (B[2,k1]) -- wait, B row 2 is index 2 in m.
        # V row index: 2 * p + k0, 2 * p + k1
        
        # P1: A[0,2] * B[2,k0] -> C[0,k0]
        U[0 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # P2: A[0,2] * B[2,k1] -> C[0,k1]
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # P3: A[1,2] * B[2,k0] -> C[1,k0]
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P4: A[1,2] * B[2,k1] -> C[1,k1]
        U[1 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
