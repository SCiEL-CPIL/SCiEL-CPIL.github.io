import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # Rank 49 for 4x4x4 part (columns 0-3 of B)
    # Rank 48 for 4x4x3 part (columns 4-6 of B)
    rank = 49 + 48
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    # Strassen 2x2x2 matrices
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 0, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 1]
    ]
    
    # Generate 49 terms for 4x4x4
    # A indices: i1 i2 j1 j2 -> i = 2*i1 + i2, j = 2*j1 + j2
    # B indices: j1 j2 k1 k2 -> j = 2*j1 + j2, k = 2*k1 + k2 (k in 0..3)
    # C indices: k1 k2 i1 i2 -> k = 2*k1 + k2, i = 2*i1 + i2
    
    col = 0
    for r1 in range(7):
        for r2 in range(7):
            r = col
            col += 1
            # U part: A_{i1 i2 j1 j2}
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u_val = U2[i1*2 + j1][r1] * U2[i2*2 + j2][r2]
                            if u_val != 0:
                                i = 2*i1 + i2
                                j = 2*j1 + j2
                                U[i*m + j][r] += u_val
            # V part: B_{j1 j2 k1 k2}
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            v_val = V2[j1*2 + k1][r1] * V2[j2*2 + k2][r2]
                            if v_val != 0:
                                j = 2*j1 + j2
                                k = 2*k1 + k2 # k is 0..3
                                V[j*p + k][r] += v_val
            # W part: C_{k1 k2 i1 i2}
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            w_val = W2[k1*2 + i1][r1] * W2[k2*2 + i2][r2]
                            if w_val != 0:
                                k = 2*k1 + k2 # k is 0..3
                                i = 2*i1 + i2
                                W[k*n + i][r] += w_val

    # --- Part 2: 4x4x3 Schoolbook (Rank 48) ---
    # B columns 4, 5, 6
    # A is 4x4, B is 4x3, C is 4x3
    for i in range(n):
        for j in range(m):
            for k in range(3):
                k_global = 4 + k # 4, 5, 6
                r = col
                col += 1
                U[i*m + j][r] = 1
                V[j*p + k_global][r] = 1
                W[k_global*n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
