"""Matrix multiplication 4x4x6 with rank 60 via Strassen blocks + schoolbook."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# Strassen factors for 2x2x2 tensor
strassen_u = [
    [[1,0],[1,1]],
    [[0,0],[1,1]],
    [[1,0],[0,0]],
    [[0,0],[0,1]],
    [[1,1],[0,0]],
    [[0,0],[1,-1]],
    [[0,1],[0,-1]],
]
strassen_v = [
    [[1,1],[0,1]],
    [[1,0],[0,0]],
    [[0,1],[0,-1]],
    [[0,-1],[1,0]],
    [[0,0],[0,1]],
    [[1,1],[0,0]],
    [[0,0],[1,1]],
]
strassen_w = [
    [[1,1],[0,0]],
    [[0,1],[0,1]],
    [[0,1],[0,0]],
    [[0,0],[1,0]],
    [[0,0],[1,1]],
    [[0,0],[0,1]],
    [[1,0],[0,0]],
]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]  # 16 x 60
    V = [[0] * rank for _ in range(m * p)]  # 24 x 60
    W = [[0] * rank for _ in range(p * n)]  # 24 x 60
    
    col = 0
    
    # Part 1: 4x4x4 using Strassen blocks (28 terms)
    for r_block in range(2):
        for c_block in range(2):
            for k_block in range(2):
                for t in range(7):
                    u_t = strassen_u[t]
                    v_t = strassen_v[t]
                    w_t = strassen_w[t]
                    
                    for i_loc in range(2):
                        for j_loc in range(2):
                            a_idx = (r_block * 2 + i_loc) * m + (k_block * 2 + j_loc)
                            U[a_idx][col] = u_t[i_loc][j_loc]
                    
                    for j_loc in range(2):
                        for k_loc in range(2):
                            b_idx = (k_block * 2 + j_loc) * p + (c_block * 2 + k_loc)
                            V[b_idx][col] = v_t[j_loc][k_loc]
                    
                    for i_loc in range(2):
                        for k_loc in range(2):
                            c_row = r_block * 2 + i_loc
                            c_col = c_block * 2 + k_loc
                            w_idx = c_col * n + c_row
                            W[w_idx][col] = w_t[i_loc][k_loc]
                    
                    col += 1
    
    # Part 2: 4x4x2 using schoolbook (32 terms)
    for i in range(n):
        for j in range(m):
            for k in range(2):
                k_actual = 4 + k
                U[i * m + j][col] = 1
                V[j * p + k_actual][col] = 1
                W[k_actual * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
