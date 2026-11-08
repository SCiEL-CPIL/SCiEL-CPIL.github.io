import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # Rank 56
    rank = 56
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # u_s, v_s, w_s are 2x2 matrices
    # s=0: M1
    u0 = [[1, 0], [0, 1]]
    v0 = [[1, 0], [0, 1]]
    w0 = [[1, 0], [0, 1]]
    
    # s=1: M2
    u1 = [[0, 0], [1, 1]]
    v1 = [[1, 0], [0, 0]]
    w1 = [[0, 0], [1, -1]]
    
    # s=2: M3
    u2 = [[1, 0], [0, 0]]
    v2 = [[0, 0], [1, -1]]
    w2 = [[0, 1], [0, 1]]
    
    # s=3: M4
    u3 = [[0, 0], [0, 1]]
    v3 = [[0, 1], [-1, 0]]
    w3 = [[1, 0], [1, 0]]
    
    # s=4: M5
    u4 = [[1, 0], [1, 0]]
    v4 = [[0, 0], [0, 1]]
    w4 = [[-1, 1], [0, 0]]
    
    # s=5: M6
    u5 = [[-1, 0], [1, 1]]
    v5 = [[0, 0], [1, 0]]
    w5 = [[0, 0], [0, 1]]
    
    # s=6: M7
    u6 = [[-1, 1], [0, 1]]
    v6 = [[0, 1], [0, 0]]
    w6 = [[1, 0], [0, 0]]
    
    strassen_u = [u0, u1, u2, u3, u4, u5, u6]
    strassen_v = [v0, v1, v2, v3, v4, v5, v6]
    strassen_w = [w0, w1, w2, w3, w4, w5, w6]
    
    # 8 blocks
    for b in range(8):
        if b < 4:
            # P_b = A_L (cols 0,1)
            # Q_b = B_T,b (rows 0,1, cols 2b, 2b+1)
            # R_b = C cols 2b, 2b+1
            j_offset = 0
            k_offset = 2 * b
            row_offset_b = 0
        else:
            # P_b = A_R (cols 2,3)
            # Q_b = B_B,b-4 (rows 2,3, cols 2(b-4), 2(b-4)+1)
            # R_b = C cols 2(b-4), 2(b-4)+1
            b_prime = b - 4
            j_offset = 2
            k_offset = 2 * b_prime
            row_offset_b = 2
            
        for s in range(7):
            c = 7 * b + s
            
            # Fill U
            # U index: i * m + j
            # i in 0,1. j in j_offset, j_offset+1
            for i_prime in range(2):
                for j_prime in range(2):
                    i_global = i_prime
                    j_global = j_offset + j_prime
                    val = strassen_u[s][i_prime][j_prime]
                    U[i_global * m + j_global][c] = val
                    
            # Fill V
            # V index: j * p + k
            # j in row_offset_b, row_offset_b+1
            # k in k_offset, k_offset+1
            for j_prime in range(2):
                for k_prime in range(2):
                    j_global = row_offset_b + j_prime
                    k_global = k_offset + k_prime
                    val = strassen_v[s][j_prime][k_prime]
                    V[j_global * p + k_global][c] = val
                    
            # Fill W
            # W index: k * n + i
            # k in k_offset, k_offset+1
            # i in 0,1
            for k_prime in range(2):
                for i_prime in range(2):
                    k_global = k_offset + k_prime
                    i_global = i_prime
                    val = strassen_w[s][i_prime][k_prime]
                    W[k_global * n + i_global][c] = val
                    
    return {"U": U, "V": V, "W": W}
