"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    n, m, p = DIMENSIONS  # 3, 3, 6
    # Output index convention: W[k*n + i, r]
    # Input A: n x m, index i*m + j
    # Input B: m x p, index j*p + k

    # Block decomposition:
    # A = [[A11 (2x2), A12 (2x1)],
    #      [A21 (1x2), A22 (1x1)]]
    # B = [[B11 (2x4), B12 (2x2)],
    #      [B21 (1x4), B22 (1x2)]]
    
    # We compute C = A * B = [[C11, C12], [C21, C22]]
    # C11 = A11 * B11
    # C12 = A11 * B12
    # C21 = A21 * B21 + A22 * B11  -- WAIT. C21 is A21 * B21? No.
    # Let's check indices.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # A is 3x3. Rows 0,1,2. Cols 0,1,2.
    # B is 3x6. Rows 0,1,2. Cols 0..5.
    #
    # Let's split A rows into {0,1} and {2}.
    # Let's split A cols into {0,1} and {2}.
    # Let's split B rows into {0,1} and {2}.
    # Let's split B cols into {0,1,2,3} and {4,5}.
    
    # A11 = A[0:2, 0:2] (2x2)
    # A12 = A[0:2, 2:3] (2x1)
    # A21 = A[2:3, 0:2] (1x2)
    # A22 = A[2:3, 2:3] (1x1)
    
    # B11 = B[0:2, 0:4] (2x4)
    # B12 = B[0:2, 4:6] (2x2)
    # B21 = B[2:3, 0:4] (1x4)
    # B22 = B[2:3, 4:6] (1x2)
    
    # C11 = A11 * B11 (2x2 * 2x4 -> 2x4)
    # C12 = A11 * B12 (2x2 * 2x2 -> 2x2)
    # C21 = A21 * B21 (1x2 * 1x4 -> 1x4)
    # C22 = A21 * B22 (1x2 * 1x2 -> 1x2)
    
    # Wait, is that all?
    # C[i,k] = sum_j A[i,j] B[j,k]
    # For i in 0,1: sum_{j=0}^2 A[i,j] B[j,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]
    #           = (A11 part) + A[i,2]B[2,k]
    # So C11 = A11 B11 + A12 B21
    #    C12 = A11 B12 + A12 B22
    # For i=2: C[2,k] = A[2,0]B[0,k] + A[2,1]B[1,k] + A[2,2]B[2,k]
    #           = A21 B11 + A22 B21
    # So C21 = A21 B11 + A22 B21
    #    C22 = A21 B12 + A22 B22
    
    # Let's list the terms:
    # 1. A11 * B11 (2x2 * 2x4). Can be decomposed as A11 * B11a + A11 * B11b where B11a, B11b are 2x2.
    #    Each 2x2 * 2x2 is rank 7 (Strassen). Total 14.
    # 2. A11 * B12 (2x2 * 2x2). Rank 7.
    # 3. A12 * B21 (2x1 * 1x4). Rank 8 (schoolbook).
    # 4. A12 * B22 (2x1 * 1x2). Rank 4 (schoolbook).
    # 5. A21 * B11 (1x2 * 2x4). Rank 8 (schoolbook).
    # 6. A21 * B12 (1x2 * 2x2). Rank 4 (schoolbook).
    # 7. A22 * B21 (1x1 * 1x4). Rank 4 (schoolbook).
    # 8. A22 * B22 (1x1 * 1x2). Rank 2 (schoolbook).
    
    # Total Rank = 14 + 7 + 8 + 4 + 8 + 4 + 4 + 2 = 51.
    
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen decomposition for X (2x2) * Y (2x2)
    # X indices in A: row r, col c
    # Y indices in B: row r', col c'
    # Z indices in C: row r, col c' (mapped to k*n+i)
    def add_strassen(X_rows, X_cols, Y_rows, Y_cols, Z_rows, Z_cols, col_offset):
        nonlocal col
        # Strassen formulas for C = X * Y
        # M1 = (X11 + X22)(Y11 + Y22)
        # M2 = (X21 + X22)Y11
        # M3 = X11(Y12 - Y22)
        # M4 = X22(Y21 - Y11)
        # M5 = (X11 + X12)Y22
        # M6 = (X21 - X11)(Y11 + Y12)
        # M7 = (X12 - X22)(Y21 + Y22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        x00, x01 = X_rows[0], X_rows[1] # Wait, X is 2x2.
        # X_rows should be [i0, i1], X_cols [j0, j1]
        x00_idx = x00 * m + X_cols[0]
        x01_idx = x00 * m + X_cols[1]
        x10_idx = x01 * m + X_cols[0]
        x11_idx = x01 * m + X_cols[1]
        
        y00_idx = Y_rows[0] * p + Y_cols[0]
        y01_idx = Y_rows[0] * p + Y_cols[1]
        y10_idx = Y_rows[1] * p + Y_cols[0]
        y11_idx = Y_rows[1] * p + Y_cols[1]
        
        z00_idx = Z_rows[0] * n + Z_cols[0] # k*n+i
        z01_idx = Z_rows[0] * n + Z_cols[1]
        z10_idx = Z_rows[1] * n + Z_cols[0]
        z11_idx = Z_rows[1] * n + Z_cols[1]
        
        # M1
        U[x00_idx][col] = 1
        U[x11_idx][col] = 1
        V[y00_idx][col] = 1
        V[y11_idx][col] = 1
        W[z00_idx][col] = 1
        W[z11_idx][col] = 1
        col += 1
        
        # M2
        U[x10_idx][col] = 1
        U[x11_idx][col] = 1
        V[y00_idx][col] = 1
        W[z10_idx][col] = 1
        W[z11_idx][col] = -1
        col += 1
        
        # M3
        U[x00_idx][col] = 1
        V[y01_idx][col] = 1
        V[y11_idx][col] = -1
        W[z01_idx][col] = 1
        W[z11_idx][col] = 1
        col += 1
        
        # M4
        U[x11_idx][col] = 1
        V[y10_idx][col] = 1
        V[y00_idx][col] = -1
        W[z00_idx][col] = 1
        W[z10_idx][col] = 1
        col += 1
        
        # M5
        U[x00_idx][col] = 1
        U[x01_idx][col] = 1
        V[y11_idx][col] = 1
        W[z00_idx][col] = -1
        W[z01_idx][col] = 1
        col += 1
        
        # M6
        U[x10_idx][col] = 1
        U[x00_idx][col] = -1
        V[y00_idx][col] = 1
        V[y01_idx][col] = 1
        W[z11_idx][col] = 1
        col += 1
        
        # M7
        U[x01_idx][col] = 1
        U[x11_idx][col] = -1
        V[y10_idx][col] = 1
        V[y11_idx][col] = 1
        W[z00_idx][col] = 1
        W[z10_idx][col] = -1
        col += 1
        
    # Helper for schoolbook
    def add_schoolbook(X_rows, X_cols, Y_rows, Y_cols, Z_rows, Z_cols, col_offset):
        nonlocal col
        # X is len(X_rows) x len(X_cols)
        # Y is len(Y_rows) x len(Y_cols)
        # Z is len(Z_rows) x len(Z_cols)
        # Z[i,j] = sum_k X[i,k] Y[k,j]
        # Here X_rows are row indices in A, X_cols are col indices in A
        # Y_rows are row indices in B, Y_cols are col indices in B
        # Z_rows are row indices in C, Z_cols are col indices in C
        
        # Note: X_cols must match Y_rows in count and order?
        # Actually, the contraction index is the columns of X and rows of Y.
        # So X_cols should correspond to Y_rows.
        # Let's assume X_cols[i] == Y_rows[i] for all i.
        
        for r_idx, i in enumerate(X_rows):
            for c_idx, k in enumerate(Z_cols):
                # Z[i, k] gets contribution from sum_j A[i, j] B[j, k]
                # j runs over the shared dimension.
                for j_idx, j in enumerate(Y_rows):
                    u_idx = i * m + X_cols[j_idx]
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
    
    # 1. A11 * B11a (B11a is B[0:2, 0:2])
    # X = A[0:2, 0:2], Y = B[0:2, 0:2], Z = C[0:2, 0:2]
    add_strassen([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], col)
    # 1b. A11 * B11b (B11b is B[0:2, 2:4])
    # X = A[0:2, 0:2], Y = B[0:2, 2:4], Z = C[0:2, 2:4]
    add_strassen([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3], col)
    
    # 2. A11 * B12 (B12 is B[0:2, 4:6])
    # X = A[0:2, 0:2], Y = B[0:2, 4:6], Z = C[0:2, 4:6]
    add_strassen([0, 1], [0, 1], [0, 1], [4, 5], [0, 1], [4, 5], col)
    
    # 3. A12 * B21 (A12 is A[0:2, 2], B21 is B[2, 0:4])
    # X = A[0:2, 2], Y = B[2, 0:4], Z = C[0:2, 0:4]
    # X_rows=[0,1], X_cols=[2]
    # Y_rows=[2], Y_cols=[0,1,2,3]
    # Z_rows=[0,1], Z_cols=[0,1,2,3]
    add_schoolbook([0, 1], [2], [2], [0, 1, 2, 3], [0, 1], [0, 1, 2, 3], col)
    
    # 4. A12 * B22 (A12 is A[0:2, 2], B22 is B[2, 4:6])
    # X = A[0:2, 2], Y = B[2, 4:6], Z = C[0:2, 4:6]
    add_schoolbook([0, 1], [2], [2], [4, 5], [0, 1], [4, 5], col)
    
    # 5. A21 * B11 (A21 is A[2, 0:2], B11 is B[0:2, 0:4])
    # X = A[2, 0:2], Y = B[0:2, 0:4], Z = C[2, 0:4]
    add_schoolbook([2], [0, 1], [0, 1], [0, 1, 2, 3], [2], [0, 1, 2, 3], col)
    
    # 6. A21 * B12 (A21 is A[2, 0:2], B12 is B[0:2, 4:6])
    # X = A[2, 0:2], Y = B[0:2, 4:6], Z = C[2, 4:6]
    add_schoolbook([2], [0, 1], [0, 1], [4, 5], [2], [4, 5], col)
    
    # 7. A22 * B21 (A22 is A[2, 2], B21 is B[2, 0:4])
    # X = A[2, 2], Y = B[2, 0:4], Z = C[2, 0:4]
    add_schoolbook([2], [2], [2], [0, 1, 2, 3], [2], [0, 1, 2, 3], col)
    
    # 8. A22 * B22 (A22 is A[2, 2], B22 is B[2, 4:6])
    # X = A[2, 2], Y = B[2, 4:6], Z = C[2, 4:6]
    add_schoolbook([2], [2], [2], [4, 5], [2], [4, 5], col)
    
    # Check if we filled all columns
    # Expected rank 51.
    if col != rank:
        # Fallback to schoolbook if logic error
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
