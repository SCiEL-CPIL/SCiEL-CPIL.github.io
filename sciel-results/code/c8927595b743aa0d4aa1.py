import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 48
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1):
        nonlocal col
        
        # Term 1
        U[a_r0 * m + a_c0][col] += 1
        U[a_r1 * m + a_c1][col] += 1
        V[b_r0 * p + b_c1][col] += 1
        V[b_r1 * p + b_c1][col] += 1
        W[c_c0 * n + c_r0][col] += 1
        W[c_c1 * n + c_r1][col] += 1
        col += 1
        
        # Term 2
        U[a_r1 * m + a_c0][col] += 1
        U[a_r1 * m + a_c1][col] += 1
        V[b_r0 * p + b_c0][col] += 1
        W[c_c0 * n + c_r1][col] += 1
        col += 1
        
        # Term 3
        U[a_r0 * m + a_c0][col] += 1
        V[b_r1 * p + b_c0][col] += 1
        V[b_r1 * p + b_c1][col] += -1
        W[c_c1 * n + c_r0][col] += 1
        W[c_c1 * n + c_r1][col] += 1
        col += 1
        
        # Term 4
        U[a_r1 * m + a_c1][col] += 1
        V[b_r0 * p + b_c0][col] += 1
        V[b_r1 * p + b_c0][col] += -1
        W[c_c0 * n + c_r0][col] += 1
        W[c_c0 * n + c_r1][col] += 1
        col += 1
        
        # Term 5
        U[a_r0 * m + a_c0][col] += 1
        U[a_r1 * m + a_c0][col] += 1
        V[b_r1 * p + b_c1][col] += 1
        W[c_c0 * n + c_r0][col] += -1
        W[c_c1 * n + c_r0][col] += 1
        col += 1
        
        # Term 6
        U[a_r0 * m + a_c0][col] += -1
        U[a_r1 * m + a_c0][col] += 1
        V[b_r0 * p + b_c0][col] += 1
        V[b_r0 * p + b_c1][col] += 1
        W[c_c1 * n + c_r1][col] += 1
        col += 1
        
        # Term 7
        U[a_r0 * m + a_c1][col] += 1
        U[a_r1 * m + a_c1][col] += -1
        V[b_r1 * p + b_c0][col] += 1
        V[b_r1 * p + b_c1][col] += 1
        W[c_c0 * n + c_r0][col] += 1
        col += 1

    def add_rank1(a_r, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r, c_c0, c_c1):
        nonlocal col
        U[a_r * m + a_c0][col] += 1
        U[a_r * m + a_c1][col] += 1
        V[b_r0 * p + b_c0][col] += 1
        V[b_r0 * p + b_c1][col] += 1
        V[b_r1 * p + b_c0][col] += 1
        V[b_r1 * p + b_c1][col] += 1
        W[c_c0 * n + c_r][col] += 1
        W[c_c1 * n + c_r][col] += 1
        col += 1

    # Strassen blocks
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1)
    
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3)
    # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3)
    
    # A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    add_strassen(0, 1, 0, 1, 0, 1, 4, 5, 0, 1, 4, 5)
    # A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    add_strassen(0, 1, 2, 3, 2, 3, 4, 5, 0, 1, 4, 5)
    
    # Rank 1 blocks
    # A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    add_rank1(2, 0, 1, 0, 1, 0, 1, 2, 0, 1)
    # A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    add_rank1(2, 2, 3, 2, 3, 0, 1, 2, 0, 1)
    
    # A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    add_rank1(2, 0, 1, 0, 1, 2, 3, 2, 2, 3)
    # A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    add_rank1(2, 2, 3, 2, 3, 2, 3, 2, 2, 3)
    
    # A[2:3, 0:2] * B[0:2, 4:6] -> C[2:3, 4:6]
    add_rank1(2, 0, 1, 0, 1, 4, 5, 2, 4, 5)
    # A[2:3, 2:4] * B[2:4, 4:6] -> C[2:3, 4:6]
    add_rank1(2, 2, 3, 2, 3, 4, 5, 2, 4, 5)

    return {"U": U, "V": V, "W": W}
