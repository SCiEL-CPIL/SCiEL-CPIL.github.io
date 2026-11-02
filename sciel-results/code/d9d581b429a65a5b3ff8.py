import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We use a block decomposition.
    # A1: 4x4 (top-left of A)
    # A2: 4x1 (top-right of A)
    # B1: 4x4 (top-left of B)
    # B2: 4x1 (top-right of B)
    # B3: 1x4 (bottom-left of B)
    # B4: 1x1 (bottom-right of B)
    
    # C = A1*B1 + A2*B3 (first 4 cols)
    # C_last = A1*B2 + A2*B4 (last col)
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. A1 * B1 using 4 blocks of 2x2 Strassen
    # A1 indices in A: i in 0..3, j in 0..3
    # B1 indices in B: j in 0..3, k in 0..3
    # C indices in C: i in 0..3, k in 0..3
    
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We have 4 blocks of 2x2 in A1 and B1.
    # Block (0,0): A1[0:2, 0:2], B1[0:2, 0:2] -> C[0:2, 0:2]
    # Block (0,1): A1[0:2, 2:4], B1[2:4, 0:2] -> C[0:2, 2:4]
    # Block (1,0): A1[2:4, 0:2], B1[0:2, 2:4] -> C[2:4, 0:2]
    # Block (1,1): A1[2:4, 2:4], B1[2:4, 2:4] -> C[2:4, 2:4]
    
    def add_strassen_block(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, base_col):
        nonlocal col
        # M1
        U[a_row_start * m + a_col_start][base_col] = 1
        U[(a_row_start+1) * m + a_col_start][base_col] = 1
        U[a_row_start * m + (a_col_start+1)][base_col] = 1
        U[(a_row_start+1) * m + (a_col_start+1)][base_col] = 1
        V[b_row_start * p + b_col_start][base_col] = 1
        V[(b_row_start+1) * p + b_col_start][base_col] = 1
        V[b_row_start * p + (b_col_start+1)][base_col] = 1
        V[(b_row_start+1) * p + (b_col_start+1)][base_col] = 1
        W[c_col_start * n + c_row_start][base_col] = 1
        W[(c_col_start+1) * n + c_row_start][base_col] = 1
        W[c_col_start * n + (c_row_start+1)][base_col] = 1
        W[(c_col_start+1) * n + (c_row_start+1)][base_col] = 1
        col += 1
        
        # M2
        U[(a_row_start+1) * m + a_col_start][base_col] = 1
        U[(a_row_start+1) * m + (a_col_start+1)][base_col] = 1
        V[b_row_start * p + b_col_start][base_col] = 1
        W[c_col_start * n + (c_row_start+1)][base_col] = 1
        W[(c_col_start+1) * n + (c_row_start+1)][base_col] = 1
        col += 1
        
        # M3
        U[a_row_start * m + a_col_start][base_col] = 1
        U[a_row_start * m + (a_col_start+1)][base_col] = 1
        V[b_row_start * p + (b_col_start+1)][base_col] = 1
        V[(b_row_start+1) * p + (b_col_start+1)][base_col] = 1
        W[c_col_start * n + c_row_start][base_col] = 1
        W[(c_col_start+1) * n + c_row_start][base_col] = 1
        col += 1
        
        # M4
        U[(a_row_start+1) * m + (a_col_start+1)][base_col] = 1
        V[(b_row_start+1) * p + b_col_start][base_col] = 1
        V[b_row_start * p + b_col_start][base_col] = 1
        W[c_col_start * n + (c_row_start+1)][base_col] = 1
        W[(c_col_start+1) * n + (c_row_start+1)][base_col] = 1
        col += 1
        
        # M5
        U[a_row_start * m + a_col_start][base_col] = 1
        U[a_row_start * m + (a_col_start+1)][base_col] = 1
        V[(b_row_start+1) * p + (b_col_start+1)][base_col] = 1
        W[c_col_start * n + c_row_start][base_col] = 1
        W[c_col_start * n + (c_row_start+1)][base_col] = 1
        col += 1
        
        # M6
        U[(a_row_start+1) * m + a_col_start][base_col] = 1
        U[a_row_start * m + a_col_start][base_col] = 1
        U[(a_row_start+1) * m + (a_col_start+1)][base_col] = 1
        U[a_row_start * m + (a_col_start+1)][base_col] = 1
        V[b_row_start * p + b_col_start][base_col] = 1
        V[b_row_start * p + (b_col_start+1)][base_col] = 1
        W[(c_col_start+1) * n + (c_row_start+1)][base_col] = 1
        col += 1
        
        # M7
        U[a_row_start * m + (a_col_start+1)][base_col] = 1
        U[(a_row_start+1) * m + (a_col_start+1)][base_col] = 1
        V[(b_row_start+1) * p + b_col_start][base_col] = 1
        V[(b_row_start+1) * p + (b_col_start+1)][base_col] = 1
        W[c_col_start * n + c_row_start][base_col] = 1
        W[(c_col_start+1) * n + (c_row_start+1)][base_col] = 1
        col += 1

    # Block (0,0)
    add_strassen_block(0, 0, 0, 0, 0, 0, col)
    # Block (0,1)
    add_strassen_block(0, 2, 2, 0, 0, 2, col)
    # Block (1,0)
    add_strassen_block(2, 0, 0, 2, 2, 0, col)
    # Block (1,1)
    add_strassen_block(2, 2, 2, 2, 2, 2, col)
    
    # 2. A2 * B3 (4 terms)
    # A2: i in 0..3, j=4
    # B3: j=4, k in 0..3
    # C: i in 0..3, k in 0..3
    for i in range(4):
        for k in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # 3. A1 * B2 (4 terms)
    # A1: i in 0..3, j in 0..3
    # B2: j in 0..3, k=4
    # C: i in 0..3, k=4
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    # 4. A2 * B4 (4 terms)
    # A2: i in 0..3, j=4
    # B4: j=4, k=4
    # C: i in 0..3, k=4
    for i in range(4):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
