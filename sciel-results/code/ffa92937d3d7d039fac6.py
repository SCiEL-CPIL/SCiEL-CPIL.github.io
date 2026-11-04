import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split B into B1 (3x4) and B2 (3x1)
    # C1 = A * B1 (3x4)
    # C2 = A * B2 (3x1)
    
    # We will use a block decomposition for C1
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B1 = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x2, B21 is 1x2, B22 is 1x2
    
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x2)
    # C21 = A21*B11 + A22*B21 (1x2)
    # C22 = A21*B12 + A22*B22 (1x2)
    
    # Ranks:
    # A11*B11: 2x2x2 -> Strassen Rank 7
    # A11*B12: 2x2x2 -> Strassen Rank 7
    # A12*B21: 2x1x2 -> Rank 4
    # A12*B22: 2x1x2 -> Rank 4
    # A21*B11: 1x2x2 -> Rank 4
    # A21*B12: 1x2x2 -> Rank 4
    # A22*B21: 1x1x2 -> Rank 2
    # A22*B22: 1x1x2 -> Rank 2
    # Total C1 Rank = 7+7+4+4+4+4+2+2 = 34
    
    # C2 = A * B2 (3x1) -> Schoolbook Rank 9
    
    # Total Rank = 34 + 9 = 43
    
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    # Inputs: A indices (2x2), B indices (2x2), C indices (2x2)
    # A is 3x3, B is 3x5, C is 5x3 (transposed output convention? No, W[k*n+i, r])
    # W index: k*n + i. C is p x n.
    # C1 is 4 cols (k=0..3), C2 is 1 col (k=4).
    
    def add_strassen_2x2x2(a_start, b_start, c_start, col_offset):
        # a_start: (i, j) for top-left of A block
        # b_start: (j, k) for top-left of B block
        # c_start: (k, i) for top-left of C block (W index k*n+i)
        nonlocal col
        # Strassen 7 terms
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # A indices:
        # A11: (i, j), A12: (i, j+1)
        # A21: (i+1, j), A22: (i+1, j+1)
        
        # B indices:
        # B11: (j, k), B12: (j, k+1)
        # B21: (j+1, k), B22: (j+1, k+1)
        
        # C indices:
        # C11: (k, i), C12: (k, i+1)
        # C21: (k+1, i), C22: (k+1, i+1)
        
        i, j = a_start
        k, _ = b_start # b_start is (j, k)
        
        # Term 1: (A11+A22)(B11+B22)
        U[i*m + j][col] += 1
        U[(i+1)*m + (j+1)][col] += 1
        V[j*p + k][col] += 1
        V[(j+1)*p + (k+1)][col] += 1
        W[k*n + i][col] += 1
        W[(k+1)*n + (i+1)][col] += 1
        col += 1
        
        # Term 2: (A21+A22)B11
        U[(i+1)*m + j][col] += 1
        U[(i+1)*m + (j+1)][col] += 1
        V[j*p + k][col] += 1
        W[k*n + (i+1)][col] += 1
        col += 1
        
        # Term 3: A11(B21-B11)
        U[i*m + j][col] += 1
        V[(j+1)*p + k][col] += 1
        V[j*p + k][col] -= 1
        W[k*n + i][col] += 1
        col += 1
        
        # Term 4: A22(B12-B22)
        U[(i+1)*m + (j+1)][col] += 1
        V[j*p + (k+1)][col] += 1
        V[(j+1)*p + (k+1)][col] -= 1
        W[(k+1)*n + (i+1)][col] += 1
        col += 1
        
        # Term 5: (A11+A12)B22
        U[i*m + j][col] += 1
        U[i*m + (j+1)][col] += 1
        V[(j+1)*p + (k+1)][col] += 1
        W[(k+1)*n + i][col] += 1
        col += 1
        
        # Term 6: (A21-A11)(B11+B12)
        U[(i+1)*m + j][col] += 1
        U[i*m + j][col] -= 1
        V[j*p + k][col] += 1
        V[j*p + (k+1)][col] += 1
        W[k*n + (i+1)][col] += 1
        W[(k+1)*n + (i+1)][col] += 1
        col += 1
        
        # Term 7: (A12-A22)(B21+B22)
        U[i*m + (j+1)][col] += 1
        U[(i+1)*m + (j+1)][col] -= 1
        V[(j+1)*p + k][col] += 1
        V[(j+1)*p + (k+1)][col] += 1
        W[k*n + i][col] += 1
        W[(k+1)*n + i][col] += 1
        col += 1
        
    def add_schoolbook_block(a_i, a_j, b_j, b_k, c_k, c_i, count=1):
        # Simple rank 1 term
        nonlocal col
        U[a_i*m + a_j][col] += 1
        V[b_j*p + b_k][col] += 1
        W[c_k*n + c_i][col] += 1
        col += 1

    # 1. A11*B11 (2x2x2) Strassen
    # A11 is at (0,0), B11 is at (0,0), C11 is at (0,0)
    add_strassen_2x2x2((0,0), (0,0), (0,0), col)
    
    # 2. A11*B12 (2x2x2) Strassen
    # A11 is at (0,0), B12 is at (0,1), C12 is at (1,0)
    add_strassen_2x2x2((0,0), (0,1), (1,0), col)
    
    # 3. A12*B21 (2x1x2) Schoolbook
    # A12: i=0,1; j=1. B21: j=1; k=0,1. C: k=0,1; i=0,1.
    # Terms:
    # (0,1) * (1,0) -> (0,0)
    add_schoolbook_block(0, 1, 1, 0, 0, 0)
    # (0,1) * (1,1) -> (1,0)
    add_schoolbook_block(0, 1, 1, 1, 1, 0)
    # (1,1) * (1,0) -> (0,1)
    add_schoolbook_block(1, 1, 1, 0, 0, 1)
    # (1,1) * (1,1) -> (1,1)
    add_schoolbook_block(1, 1, 1, 1, 1, 1)
    
    # 4. A12*B22 (2x1x2) Schoolbook
    # A12: i=0,1; j=1. B22: j=1; k=2,3. C: k=2,3; i=0,1.
    add_schoolbook_block(0, 1, 1, 2, 2, 0)
    add_schoolbook_block(0, 1, 1, 3, 3, 0)
    add_schoolbook_block(1, 1, 1, 2, 2, 1)
    add_schoolbook_block(1, 1, 1, 3, 3, 1)
    
    # 5. A21*B11 (1x2x2) Schoolbook
    # A21: i=2; j=0,1. B11: j=0,1; k=0,1. C: k=0,1; i=2.
    add_schoolbook_block(2, 0, 0, 0, 0, 2)
    add_schoolbook_block(2, 0, 1, 0, 1, 2)
    add_schoolbook_block(2, 1, 0, 0, 0, 2)
    add_schoolbook_block(2, 1, 1, 0, 1, 2)
    
    # 6. A21*B12 (1x2x2) Schoolbook
    # A21: i=2; j=0,1. B12: j=0,1; k=2,3. C: k=2,3; i=2.
    add_schoolbook_block(2, 0, 0, 2, 2, 2)
    add_schoolbook_block(2, 0, 1, 2, 3, 2)
    add_schoolbook_block(2, 1, 0, 2, 2, 2)
    add_schoolbook_block(2, 1, 1, 2, 3, 2)
    
    # 7. A22*B21 (1x1x2) Schoolbook
    # A22: i=2; j=2. B21: j=2; k=0,1. C: k=0,1; i=2.
    add_schoolbook_block(2, 2, 2, 0, 0, 2)
    add_schoolbook_block(2, 2, 2, 1, 1, 2)
    
    # 8. A22*B22 (1x1x2) Schoolbook
    # A22: i=2; j=2. B22: j=2; k=2,3. C: k=2,3; i=2.
    add_schoolbook_block(2, 2, 2, 2, 2, 2)
    add_schoolbook_block(2, 2, 2, 3, 3, 2)
    
    # 9. C2 = A * B2 (3x1) Schoolbook
    # B2 is col 4.
    for i in range(3):
        for j in range(3):
            add_schoolbook_block(i, j, j, 4, 4, i)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
