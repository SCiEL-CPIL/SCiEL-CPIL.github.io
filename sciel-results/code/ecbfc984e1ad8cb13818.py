import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 2 * 4 * 7 = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 multiplication
    # u[i, j], v[j, k], w[i, k]
    # Note: w index in W is k*n + i. Here local k is k_low, i is global i.
    
    strassen_terms = [
        # M1
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        # M2
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, -1]]),
        # M3
        ([[1, 0], [0, 0]], [[0, 0], [1, -1]], [[0, 1], [0, 1]]),
        # M4
        ([[0, 0], [0, 1]], [[-1, 1], [0, 0]], [[1, 0], [1, 0]]),
        # M5
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]),
        # M6
        ([[-1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        # M7
        ([[0, 1], [-1, 1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]])
    ]
    
    col = 0
    for j_high in range(2):
        for k_high in range(4):
            for u_mat, v_mat, w_mat in strassen_terms:
                # Fill U
                for i in range(2):
                    for j_low in range(2):
                        val = u_mat[i][j_low]
                        if val != 0:
                            u_idx = i * m + (2 * j_high + j_low)
                            U[u_idx][col] = val
                            
                # Fill V
                for j_low in range(2):
                    for k_low in range(2):
                        val = v_mat[j_low][k_low]
                        if val != 0:
                            v_idx = (2 * j_high + j_low) * p + (2 * k_high + k_low)
                            V[v_idx][col] = val
                            
                # Fill W
                for i in range(2):
                    for k_low in range(2):
                        val = w_mat[i][k_low]
                        if val != 0:
                            # W index is k*n + i
                            # k is global k = 2 * k_high + k_low
                            w_idx = (2 * k_high + k_low) * n + i
                            W[w_idx][col] = val
                            
                col += 1
                
    return {"U": U, "V": V, "W": W}
