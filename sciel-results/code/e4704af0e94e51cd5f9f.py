import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_right (2x1) * B_bottom (1x7)
    # A_right is column 2 of A. Indices i=0,1; j=2.
    # B_bottom is row 2 of B. Index j=2; k=0..6.
    # Product contributes to C[i, k].
    # W index: k*n + i = 2k + i.
    
    for i in range(2):
        for k in range(7):
            U[i * 3 + 2][col] = 1
            V[2 * 7 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: A_left (2x2) * B_top (2x7)
    # A_left is cols 0,1 of A.
    # B_top is rows 0,1 of B.
    
    # Strassen blocks for columns (0,1), (2,3), (4,5)
    strassen_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for k0, k1 in strassen_blocks:
        # B indices for this block:
        # B[0, k0] -> V[0*7 + k0]
        # B[0, k1] -> V[0*7 + k1]
        # B[1, k0] -> V[1*7 + k0]
        # B[1, k1] -> V[1*7 + k1]
        
        # A indices:
        # A[0,0] -> U[0]
        # A[0,1] -> U[1]
        # A[1,0] -> U[3]
        # A[1,1] -> U[4]
        
        # W indices for C[i, k]:
        # C[0, k0] -> W[k0*2 + 0]
        # C[0, k1] -> W[k1*2 + 0]
        # C[1, k0] -> W[k0*2 + 1]
        # C[1, k1] -> W[k1*2 + 1]
        
        # M1 = (A[1,0] + A[1,1]) * (B[0,k0] + B[0,k1])
        U[3][col] = 1
        U[4][col] = 1
        V[k0][col] = 1
        V[7 + k1][col] = 1 # Wait, V index is j*p + k. j=0, k=k1 -> k1.
        # Correction: V[0*7 + k1] = V[k1].
        V[k1][col] = 1
        # W contributions: C[0,k0] += 1, C[1,k1] += 1
        W[k0 * 2 + 0][col] = 1
        W[k1 * 2 + 1][col] = 1
        col += 1
        
        # M2 = A[1,1] * (B[0,k0] + B[0,k1])
        U[4][col] = 1
        V[k0][col] = 1
        V[k1][col] = 1
        # W contributions: C[0,k1] += 1
        W[k1 * 2 + 0][col] = 1
        col += 1
        
        # M3 = (A[0,0] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[0][col] = 1
        U[1][col] = -1
        V[7 + k0][col] = 1
        V[7 + k1][col] = 1
        # W contributions: C[1,k0] += 1, C[1,k1] += 1
        W[k0 * 2 + 1][col] = 1
        W[k1 * 2 + 1][col] = 1
        col += 1
        
        # M4 = A[0,0] * (B[1,k1] - B[0,k1])
        U[0][col] = 1
        V[7 + k1][col] = 1
        V[k1][col] = -1
        # W contributions: C[0,k0] += -1, C[0,k1] += 1
        W[k0 * 2 + 0][col] = -1
        W[k1 * 2 + 0][col] = 1
        col += 1
        
        # M5 = (A[0,0] + A[1,0]) * B[1,k1]
        U[0][col] = 1
        U[3][col] = 1
        V[7 + k1][col] = 1
        # W contributions: C[0,k0] += 1, C[1,k0] += 1
        W[k0 * 2 + 0][col] = 1
        W[k0 * 2 + 1][col] = 1
        col += 1
        
        # M6 = (A[0,1] - A[1,1]) * (B[0,k0] + B[1,k0])
        U[1][col] = 1
        U[4][col] = -1
        V[k0][col] = 1
        V[7 + k0][col] = 1
        # W contributions: C[1,k1] += 1
        W[k1 * 2 + 1][col] = 1
        col += 1
        
        # M7 = (A[0,0] + A[1,0]) * (B[0,k1] - B[1,k1])
        U[0][col] = 1
        U[3][col] = 1
        V[k1][col] = 1
        V[7 + k1][col] = -1
        # W contributions: C[0,k0] += -1
        W[k0 * 2 + 0][col] = -1
        col += 1

    # Part 3: A_left (2x2) * B_col6 (2x1)
    # B_col6 is column 6 of B_top.
    # C[0,6] = A[0,0]B[0,6] + A[0,1]B[1,6]
    # C[1,6] = A[1,0]B[0,6] + A[1,1]B[1,6]
    
    # Term 1: A[0,0] * B[0,6] -> C[0,6]
    U[0][col] = 1
    V[6][col] = 1
    W[6 * 2 + 0][col] = 1
    col += 1
    
    # Term 2: A[0,1] * B[1,6] -> C[0,6]
    U[1][col] = 1
    V[7 + 6][col] = 1
    W[6 * 2 + 0][col] = 1
    col += 1
    
    # Term 3: A[1,0] * B[0,6] -> C[1,6]
    U[3][col] = 1
    V[6][col] = 1
    W[6 * 2 + 1][col] = 1
    col += 1
    
    # Term 4: A[1,1] * B[1,6] -> C[1,6]
    U[4][col] = 1
    V[7 + 6][col] = 1
    W[6 * 2 + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
