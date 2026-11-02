import time
import sys

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # alpha[s][i][j] is coefficient of A_ij in A_s
    # beta[s][i][j] is coefficient of B_ij in B_s
    # w[s][i][j] is coefficient of M_s in C_ij
    
    # Indices for blocks:
    # A: 11, 12, 21, 22
    # B: 11, 12, 21, 22
    # C: 11, 12, 21, 22
    
    # s=1: M1 = (A11+A22)(B11+B22)
    alpha = [
        {}, # 0
        { (0,0): 1, (1,1): 1 }, # 1
        { (1,0): 1, (1,1): 1 }, # 2
        { (0,0): 1 },           # 3
        { (1,1): 1 },           # 4
        { (0,0): 1, (0,1): 1 }, # 5
        { (1,0): 1, (0,0): -1 },# 6
        { (0,1): 1, (1,1): -1 } # 7
    ]
    
    beta = [
        {}, # 0
        { (0,0): 1, (1,1): 1 }, # 1
        { (0,0): 1 },           # 2
        { (0,1): 1, (1,1): -1 },# 3
        { (1,0): 1, (0,0): -1 },# 4
        { (1,1): 1 },           # 5
        { (0,0): 1, (0,1): 1 }, # 6
        { (1,0): 1, (1,1): 1 }  # 7
    ]
    
    w_coeff = [
        {}, # 0
        { (0,0): 1, (1,1): 1 }, # 1
        { (1,0): 1, (1,1): -1 },# 2
        { (0,1): 1, (1,1): 1 }, # 3
        { (0,0): 1, (1,0): 1 }, # 4
        { (0,0): -1, (0,1): 1 },# 5
        { (1,1): 1 },           # 6
        { (0,0): 1 }            # 7
    ]
    
    for s in range(1, 8):
        for t in range(4):
            c = 4 * (s - 1) + t
            
            # Determine which element of A_s is used
            # t=0,2 -> a0; t=1,3 -> a1
            k_a = 0 if t in [0, 2] else 1
            
            # Determine which element of B_s is used
            # t=0 -> b00; t=1 -> b10; t=2 -> b01; t=3 -> b11
            r_b = 1 if t in [1, 3] else 0
            c_b = 1 if t in [2, 3] else 0
            
            # Determine which element of M_s is produced
            # t=0,1 -> M0; t=2,3 -> M1
            k_c = 0 if t in [0, 1] else 1
            
            # Fill U
            for (i, j), val in alpha[s].items():
                if val != 0:
                    # A_ij is at row i, cols 2j, 2j+1
                    # We use element k_a of A_ij
                    u_idx = i * m + 2 * j + k_a
                    U[u_idx][c] += val
            
            # Fill V
            for (i, j), val in beta[s].items():
                if val != 0:
                    # B_ij is at rows 2i, 2i+1, cols 2j, 2j+1
                    # We use element (r_b, c_b) of B_ij
                    v_row = 2 * i + r_b
                    v_col = 2 * j + c_b
                    v_idx = v_row * p + v_col
                    V[v_idx][c] += val
            
            # Fill W
            for (i, j), val in w_coeff[s].items():
                if val != 0:
                    # M_s contributes to C_ij
                    # We produce element k_c of M_s
                    # C_ij is at row i, cols 2j, 2j+1
                    # We add to element k_c of C_ij
                    w_col = 2 * j + k_c
                    w_row = i
                    # W index is k*n + i, where k is col index of C
                    w_idx = w_col * n + w_row
                    W[w_idx][c] += val
    
    return {"U": U, "V": V, "W": W}
