"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 29 decomposition
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Term 1: j=1 (index 1 in 0-based j? No, j=1 is the second column)
    # A indices: i*m + j. i in 0..1, j in 0..2.
    # j=1 corresponds to indices 1 (i=0) and 4 (i=1).
    # B indices: j*p + k. j=1 corresponds to indices 8..15.
    # C indices: k*n + i.
    
    # U for term 0
    U[1][0] = 1
    U[4][0] = 1
    
    # V for term 0
    for k in range(p):
        V[1 * p + k][0] = 1
        
    # W for term 0
    for k in range(p):
        for i in range(n):
            W[k * n + i][0] = 1
            
    # Term 2: j=2,3 (indices 2,3 in 0-based j)
    # A indices: 
    # i=0, j=2 -> 2
    # i=0, j=3 -> 3
    # i=1, j=2 -> 5
    # i=1, j=3 -> 6
    # B indices:
    # j=2, k=0..7 -> 16..23
    # j=3, k=0..7 -> 24..31
    
    # 4 blocks of 2x2
    for b in range(4):
        k_base = b * 2
        col_base = 1 + b * 7
        
        # P1 = (A00 + A11)(B00 + B11)
        # U: A00 (2), A11 (6)
        U[2][col_base] = 1
        U[6][col_base] = 1
        # V: B00 (16+k), B11 (24+k)
        V[16 + k_base][col_base] = 1
        V[16 + k_base + 1][col_base] = 1
        V[24 + k_base][col_base] = 1
        V[24 + k_base + 1][col_base] = 1
        # W: C00 (+1), C11 (+1)
        W[k_base * n + 0][col_base] = 1
        W[(k_base + 1) * n + 1][col_base] = 1
        
        # P2 = (A21 + A22)B11
        # Local A21 is A[1,2] -> idx 5. Local A22 is A[1,3] -> idx 6.
        # Local B11 is B[3, k] -> idx 24+k.
        U[5][col_base + 1] = 1
        U[6][col_base + 1] = 1
        V[24 + k_base][col_base + 1] = 1
        V[24 + k_base + 1][col_base + 1] = 1
        # W: C10 (+1), C11 (-1)
        W[k_base * n + 1][col_base + 1] = 1
        W[(k_base + 1) * n + 1][col_base + 1] = -1
        
        # P3 = A11(B12 - B22)
        # Local A11 is A[0,3] -> idx 3.
        # Local B12 is B[2, k+1] -> idx 16+k+1.
        # Local B22 is B[3, k+1] -> idx 24+k+1.
        U[3][col_base + 2] = 1
        V[16 + k_base + 1][col_base + 2] = 1
        V[24 + k_base + 1][col_base + 2] = -1
        # W: C01 (+1), C11 (+1)
        W[(k_base + 1) * n + 0][col_base + 2] = 1
        W[(k_base + 1) * n + 1][col_base + 2] = 1
        
        # P4 = A22(B21 - B11)
        # Local A22 is A[1,3] -> idx 6.
        # Local B21 is B[3, k] -> idx 24+k.
        # Local B11 is B[2, k] -> idx 16+k.
        U[6][col_base + 3] = 1
        V[24 + k_base][col_base + 3] = 1
        V[16 + k_base][col_base + 3] = -1
        # W: C00 (+1), C10 (+1)
        W[k_base * n + 0][col_base + 3] = 1
        W[k_base * n + 1][col_base + 3] = 1
        
        # P5 = (A11 + A12)B22
        # Local A11 is A[0,3] -> idx 3.
        # Local A12 is A[0,2] -> idx 2.
        # Local B22 is B[3, k+1] -> idx 24+k+1.
        U[3][col_base + 4] = 1
        U[2][col_base + 4] = 1
        V[24 + k_base + 1][col_base + 4] = 1
        # W: C01 (+1), C00 (-1)
        W[(k_base + 1) * n + 0][col_base + 4] = 1
        W[k_base * n + 0][col_base + 4] = -1
        
        # P6 = (A21 - A11)(B11 + B12)
        # Local A21 is A[1,2] -> idx 5.
        # Local A11 is A[0,3] -> idx 3.
        # Local B11 is B[2, k] -> idx 16+k.
        # Local B12 is B[2, k+1] -> idx 16+k+1.
        U[5][col_base + 5] = 1
        U[3][col_base + 5] = -1
        V[16 + k_base][col_base + 5] = 1
        V[16 + k_base + 1][col_base + 5] = 1
        # W: C11 (+1)
        W[(k_base + 1) * n + 1][col_base + 5] = 1
        
        # P7 = (A12 - A22)(B21 + B22)
        # Local A12 is A[0,2] -> idx 2.
        # Local A22 is A[1,3] -> idx 6.
        # Local B21 is B[3, k] -> idx 24+k.
        # Local B22 is B[3, k+1] -> idx 24+k+1.
        U[2][col_base + 6] = 1
        U[6][col_base + 6] = -1
        V[24 + k_base][col_base + 6] = 1
        V[24 + k_base + 1][col_base + 6] = 1
        # W: C00 (+1)
        W[k_base * n + 0][col_base + 6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
