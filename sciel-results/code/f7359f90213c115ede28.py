import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's algorithm decomposition for 2x2 matrix multiplication
    # Rank 7
    
    n, m, p = DIMENSIONS
    rank = 7
    
    # U shape: (n*m, rank) = (4, 7)
    # V shape: (m*p, rank) = (4, 7)
    # W shape: (p*n, rank) = (4, 7)
    
    # Indices for U (A): 0:00, 1:01, 2:10, 3:11
    # Indices for V (B): 0:00, 1:01, 2:10, 3:11
    # Indices for W (C): W[k*n+i,r] -> 0:00, 1:10, 2:01, 3:11
    
    # Define the 7 bilinear forms M1..M7 as (U_vec, V_vec) pairs
    # U_vec is length 4 (A00, A01, A10, A11)
    # V_vec is length 4 (B00, B01, B10, B11)
    
    # M1 = (A00 + A11) * (B00 + B11)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    
    # M2 = (A10 + A11) * B00
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    
    # M3 = A00 * (B01 - B11)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    
    # M4 = A11 * (B10 - B00)
    u4 = [0, 0, 0, 1]
    v4 = [-1, 0, 1, 0]
    
    # M5 = (A00 + A01) * B11
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    
    # M6 = (A10 - A00) * (B00 + B01)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    
    # M7 = (A01 - A11) * (B10 + B11)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    
    # Construct U and V matrices
    # U[i, k] = coefficient of A[i] in M[k]
    # V[j, k] = coefficient of B[j] in M[k]
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    
    for k in range(rank):
        if k == 0:
            u_vec, v_vec = u1, v1
        elif k == 1:
            u_vec, v_vec = u2, v2
        elif k == 2:
            u_vec, v_vec = u3, v3
        elif k == 3:
            u_vec, v_vec = u4, v4
        elif k == 4:
            u_vec, v_vec = u5, v5
        elif k == 5:
            u_vec, v_vec = u6, v6
        elif k == 6:
            u_vec, v_vec = u7, v7
            
        for i in range(n*m):
            U[i][k] = u_vec[i]
        for j in range(m*p):
            V[j][k] = v_vec[j]
            
    # Construct W matrix
    # W[r, k] = coefficient of M[k] in C[r]
    # C[0,0] (idx 0) = M1 + M4 - M5 + M7
    # C[1,0] (idx 1) = M2 + M4
    # C[0,1] (idx 2) = M3 + M5
    # C[1,1] (idx 3) = M1 - M2 + M3 + M6
    
    W = [[0]*rank for _ in range(p*n)]
    
    # idx 0: C[0,0]
    W[0][0] = 1
    W[0][3] = 1
    W[0][4] = -1
    W[0][6] = 1
    
    # idx 1: C[1,0]
    W[1][1] = 1
    W[1][3] = 1
    
    # idx 2: C[0,1]
    W[2][2] = 1
    W[2][4] = 1
    
    # idx 3: C[1,1]
    W[3][0] = 1
    W[3][1] = -1
    W[3][2] = 1
    W[3][5] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
