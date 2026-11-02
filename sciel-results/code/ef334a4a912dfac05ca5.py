import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 34
    
    # Initialize U, V, W
    # U: (n*m, rank) = (9, 34)
    # V: (m*p, rank) = (12, 34)
    # W: (p*n, rank) = (12, 34)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        nonlocal col
        # A is 2x2 at A_rows[0:2], A_cols[0:2]
        # B is 2x2 at B_rows[0:2], B_cols[0:2]
        # C is 2x2 at C_rows[0:2], C_cols[0:2]
        
        # P1 = (A00 + A11) * (B00 + B11)
        U[A_rows[0]*m + A_cols[0]][col] += 1
        U[A_rows[1]*m + A_cols[1]][col] += 1
        V[B_rows[0]*p + B_cols[0]][col] += 1
        V[B_rows[1]*p + B_cols[1]][col] += 1
        # C00 += P1, C11 += P1
        W[C_cols[0]*n + C_rows[0]][col] += 1
        W[C_cols[1]*n + C_rows[1]][col] += 1
        col += 1
        
        # P2 = (A10 + A11) * B00
        U[A_rows[1]*m + A_cols[0]][col] += 1
        U[A_rows[1]*m + A_cols[1]][col] += 1
        V[B_rows[0]*p + B_cols[0]][col] += 1
        # C10 += P2
        W[C_cols[0]*n + C_rows[1]][col] += 1
        col += 1
        
        # P3 = A00 * (B01 - B11)
        U[A_rows[0]*m + A_cols[0]][col] += 1
        V[B_rows[0]*p + B_cols[1]][col] += 1
        V[B_rows[1]*p + B_cols[1]][col] -= 1
        # C01 += P3
        W[C_cols[1]*n + C_rows[0]][col] += 1
        col += 1
        
        # P4 = A11 * (B10 - B00)
        U[A_rows[1]*m + A_cols[1]][col] += 1
        V[B_rows[1]*p + B_cols[0]][col] += 1
        V[B_rows[0]*p + B_cols[0]][col] -= 1
        # C00 += P4, C10 += P4
        W[C_cols[0]*n + C_rows[0]][col] += 1
        W[C_cols[0]*n + C_rows[1]][col] += 1
        col += 1
        
        # P5 = (A00 + A01) * B11
        U[A_rows[0]*m + A_cols[0]][col] += 1
        U[A_rows[0]*m + A_cols[1]][col] += 1
        V[B_rows[1]*p + B_cols[1]][col] += 1
        # C00 -= P5, C01 += P5
        W[C_cols[0]*n + C_rows[0]][col] -= 1
        W[C_cols[1]*n + C_rows[0]][col] += 1
        col += 1
        
        # P6 = (A10 - A00) * (B00 + B01)
        U[A_rows[1]*m + A_cols[0]][col] += 1
        U[A_rows[0]*m + A_cols[0]][col] -= 1
        V[B_rows[0]*p + B_cols[0]][col] += 1
        V[B_rows[0]*p + B_cols[1]][col] += 1
        # C11 += P6
        W[C_cols[1]*n + C_rows[1]][col] += 1
        col += 1
        
        # P7 = (A01 - A11) * (B10 + B11)
        U[A_rows[0]*m + A_cols[1]][col] += 1
        U[A_rows[1]*m + A_cols[1]][col] -= 1
        V[B_rows[1]*p + B_cols[0]][col] += 1
        V[B_rows[1]*p + B_cols[1]][col] += 1
        # C00 += P7
        W[C_cols[0]*n + C_rows[0]][col] += 1
        col += 1

    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A rows 0,1 cols 0,1. B rows 0,1 cols 0,1. C rows 0,1 cols 0,1.
    add_strassen([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # 2. Strassen for A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A rows 0,1 cols 0,1. B rows 0,1 cols 2,3. C rows 0,1 cols 2,3.
    add_strassen([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # 3. Cross terms A[0:2, 2] * B[2, 0:4] -> C[0:2, 0:4]
    # A02 * B2k
    for k in range(4):
        U[0*m + 2][col] = 1
        V[2*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        U[1*m + 2][col] = 1
        V[2*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1
        
    # 4. Cross terms A[2, 0:2] * B[0:2, 0:4] -> C[2, 0:4]
    # A20 * B0k
    for k in range(4):
        U[2*m + 0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 2][col] = 1
        col += 1
        # A21 * B1k
        U[2*m + 1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 2][col] = 1
        col += 1
        
    # 5. Cross term A[2, 2] * B[2, 0:4] -> C[2, 0:4]
    # A22 * B2k
    for k in range(4):
        U[2*m + 2][col] = 1
        V[2*p + k][col] = 1
        W[k*n + 2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
