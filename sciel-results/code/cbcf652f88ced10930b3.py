DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Strategy: Split B into B1 (3x3) and B2 (3x2)
    # C = A * B1 + A * B2
    # Use Strassen (rank 7) for 2x2 block in 3x3x3, plus schoolbook for remaining
    # Total: 7 (Strassen) + 12 (schoolbook for 3x3x3 remainder) + 18 (schoolbook 3x3x2) = 37
    # Actually, let's be more careful.
    
    # For 3x3x3: split into 2x2 block + remaining
    # A = [[A11, A12], [A21, A22, A23]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1, A23 is 1x1
    # B1 = [[B11, B12], [B21, B22, B23]] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1, B23 is 1x1
    
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x1)
    # C13 = A11*B13 + A12*B23 (2x1)  -- wait, B1 is 3x3 so B13 doesn't exist
    # Let me redo: B1 is 3x3, so columns are 0,1,2
    # C11 (2x2) = A11*B11 + A12*B21
    # C12 (2x1) = A11*B12 + A12*B22
    # C13 (2x1) = A11*B13... no, B1 only has 3 columns
    # Actually for 3x3: A is 3x3, B is 3x3, C is 3x3
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let me use a simpler approach: just use schoolbook for everything but verify
    # Actually, let me implement Strassen for the 2x2 block properly.
    
    # Partition: rows 0,1 and row 2; cols 0,1 and col 2
    # A = [A11 A12; A21 A22 A23] where A11=2x2, A12=2x1, A21=1x2, A22=1x1, A23=1x1
    # B = [B11 B12; B21 B22 B23] where B11=2x2, B12=2x1, B21=1x2, B22=1x1, B23=1x1
    
    # C11 (2x2) = A11*B11 + A12*B21
    # C12 (2x1) = A11*B12 + A12*B22
    # C21 (1x2) = A21*B11 + A22*B21
    # C22 (1x1) = A21*B12 + A22*B22
    # C23 (1x1) = A23*B22... wait, that's not right
    
    # Let me just use the full schoolbook decomposition which is guaranteed correct.
    # Rank = 3*3*5 = 45
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
