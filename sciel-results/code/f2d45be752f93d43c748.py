"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 independent Strassen (rank 7) decompositions
    rank = 28
    
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (8, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We have 4 blocks of B: B11, B21, B12, B22
    # Each block uses a specific set of A columns and B rows/cols
    
    # Block 1: A1 (cols 0,1) * B11 (rows 0,1; cols 0,1) -> C cols 0,1
    # Block 2: A2 (cols 2,3) * B21 (rows 2,3; cols 0,1) -> C cols 0,1
    # Block 3: A1 (cols 0,1) * B12 (rows 0,1; cols 2,3) -> C cols 2,3
    # Block 4: A2 (cols 2,3) * B22 (rows 2,3; cols 2,3) -> C cols 2,3
    
    # Helper to add a Strassen decomposition
    # a_cols: indices of A columns (2 indices)
    # b_rows: indices of B rows (2 indices)
    # c_cols: indices of C columns (2 indices)
    # offset: starting column index in U, V, W
    
    def add_strassen(a_cols, b_rows, c_cols, offset):
        # a_cols = [j1, j2]
        # b_rows = [k1, k2]
        # c_cols = [c1, c2]
        
        # A indices in U: i*m + j
        # B indices in V: j*p + k
        # C indices in W: k*n + i
        
        # M1: (A11+A22)(B11+B22)
        # U: A11, A22
        # V: B11, B22
        # W: C11, C22
        
        # A11 is A[0, j1], A22 is A[1, j2]
        # B11 is B[j1, k1], B22 is B[j2, k2]
        
        # C11 is C[0, c1], C22 is C[1, c2]
        
        # M1
        col = offset + 0
        U[0*m + a_cols[0]][col] += 1
        U[1*m + a_cols[1]][col] += 1
        V[a_cols[0]*p + b_rows[0]][col] += 1
        V[a_cols[1]*p + b_rows[1]][col] += 1
        W[b_rows[0]*n + 0][col] += 1
        W[b_rows[1]*n + 1][col] += 1
        
        # M2: (A21+A22)B11
        # A21 is A[1, j1], A22 is A[1, j2]
        # B11 is B[j1, k1]
        # C21 is C[1, c1], C11 is C[0, c1] (part of sum)
        
        col = offset + 1
        U[1*m + a_cols[0]][col] += 1
        U[1*m + a_cols[1]][col] += 1
        V[a_cols[0]*p + b_rows[0]][col] += 1
        W[b_rows[0]*n + 1][col] += 1
        
        # M3: A11(B21-B22)
        # A11 is A[0, j1]
        # B21 is B[j2, k1], B22 is B[j2, k2]
        # C12 is C[0, c2], C22 is C[1, c2]
        
        col = offset + 2
        U[0*m + a_cols[0]][col] += 1
        V[a_cols[1]*p + b_rows[0]][col] += 1
        V[a_cols[1]*p + b_rows[1]][col] -= 1
        W[b_rows[0]*n + 0][col] += 1
        W[b_rows[1]*n + 1][col] -= 1
        
        # M4: A22(B12-B11)
        # A22 is A[1, j2]
        # B12 is B[j1, k2], B11 is B[j1, k1]
        # C22 is C[1, c2], C12 is C[0, c2]
        
        col = offset + 3
        U[1*m + a_cols[1]][col] += 1
        V[a_cols[0]*p + b_rows[1]][col] += 1
        V[a_cols[0]*p + b_rows[0]][col] -= 1
        W[b_rows[1]*n + 1][col] += 1
        W[b_rows[0]*n + 0][col] -= 1
        
        # M5: (A11+A12)B22
        # A11 is A[0, j1], A12 is A[0, j2]
        # B22 is B[j2, k2]
        # C12 is C[0, c2], C22 is C[1, c2]
        
        col = offset + 4
        U[0*m + a_cols[0]][col] += 1
        U[0*m + a_cols[1]][col] += 1
        V[a_cols[1]*p + b_rows[1]][col] += 1
        W[b_rows[1]*n + 0][col] += 1
        
        # M6: (A21-A11)(B11+B12)
        # A21 is A[1, j1], A11 is A[0, j1]
        # B11 is B[j1, k1], B12 is B[j1, k2]
        # C21 is C[1, c1], C11 is C[0, c1]
        
        col = offset + 5
        U[1*m + a_cols[0]][col] += 1
        U[0*m + a_cols[0]][col] -= 1
        V[a_cols[0]*p + b_rows[0]][col] += 1
        V[a_cols[0]*p + b_rows[1]][col] += 1
        W[b_rows[0]*n + 1][col] += 1
        W[b_rows[1]*n + 0][col] += 1
        
        # M7: (A12-A22)(B21+B22)
        # A12 is A[0, j2], A22 is A[1, j2]
        # B21 is B[j2, k1], B22 is B[j2, k2]
        # C12 is C[0, c2], C22 is C[1, c2]
        
        col = offset + 6
        U[0*m + a_cols[1]][col] += 1
        U[1*m + a_cols[1]][col] -= 1
        V[a_cols[1]*p + b_rows[0]][col] += 1
        V[a_cols[1]*p + b_rows[1]][col] += 1
        W[b_rows[0]*n + 0][col] += 1
        W[b_rows[1]*n + 1][col] += 1
        
    # Block 1: A1 (cols 0,1) * B11 (rows 0,1; cols 0,1)
    add_strassen([0, 1], [0, 1], [0, 1], 0)
    
    # Block 2: A2 (cols 2,3) * B21 (rows 2,3; cols 0,1)
    add_strassen([2, 3], [2, 3], [0, 1], 7)
    
    # Block 3: A1 (cols 0,1) * B12 (rows 0,1; cols 2,3)
    add_strassen([0, 1], [0, 1], [2, 3], 14)
    
    # Block 4: A2 (cols 2,3) * B22 (rows 2,3; cols 2,3)
    add_strassen([2, 3], [2, 3], [2, 3], 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
