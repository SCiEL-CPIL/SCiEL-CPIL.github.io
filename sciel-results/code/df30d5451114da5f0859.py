import time

def construct():
    """
    Rank-24 exact decomposition for 2x4 @ 4x6 matrix multiplication.
    
    Strategy: Split the 4x4 middle dimension into two 2x2 blocks (left and right halves).
    This gives C = A_left @ B_top + A_right @ B_bottom, where each term is a 2x2 @ 2x6 multiplication.
    Each 2x2 @ 2x6 has rank 12 (schoolbook, since each output element C[i,j] = A[i,1]*B[1,j] + A[i,2]*B[2,j]
    requires 2 multiplications, and there are 2*6=12 output elements).
    Total rank = 12 + 12 = 24.
    
    U: (n*m, rank) = (8, 24)
    V: (m*p, rank) = (24, 24)
    W: (p*n, rank) = (12, 24)
    
    W index convention: W[k*n+i, r] where k is column index (0..5), i is row index (0..1)
    So W index = k*2 + i for output C[i][k]
    """
    n, m, p = 2, 4, 6
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 12 x 24
    
    # Left block: A_left is rows 0-1, cols 0-1 of A (2x2)
    # B_top is rows 0-1, cols 0-5 of B (2x6)
    # Output: C_left[i][j] for i in 0..1, j in 0..5
    # C_left[i][j] = A[0][0]*B[0][j] + A[0][1]*B[1][j] for i=0
    #                A[1][0]*B[0][j] + A[1][1]*B[1][j] for i=1
    
    # Right block: A_right is rows 0-1, cols 2-3 of A (2x2)
    # B_bottom is rows 2-3, cols 0-5 of B (2x6)
    # Output: C_right[i][j] for i in 0..1, j in 0..5
    
    col = 0
    # Left block decomposition (rank 12)
    for i in range(n):  # i = 0, 1
        for j in range(p):  # j = 0, 1, 2, 3, 4, 5
            # Term 1: A[i][0] * B[0][j]
            U[i * m + 0][col] = 1
            V[0 * p + j][col] = 1
            W[j * n + i][col] = 1
            col += 1
            
            # Term 2: A[i][1] * B[1][j]
            U[i * m + 1][col] = 1
            V[1 * p + j][col] = 1
            W[j * n + i][col] = 1
            col += 1
    
    # Right block decomposition (rank 12)
    for i in range(n):  # i = 0, 1
        for j in range(p):  # j = 0, 1, 2, 3, 4, 5
            # Term 1: A[i][2] * B[2][j]
            U[i * m + 2][col] = 1
            V[2 * p + j][col] = 1
            W[j * n + i][col] = 1
            col += 1
            
            # Term 2: A[i][3] * B[3][j]
            U[i * m + 3][col] = 1
            V[3 * p + j][col] = 1
            W[j * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
