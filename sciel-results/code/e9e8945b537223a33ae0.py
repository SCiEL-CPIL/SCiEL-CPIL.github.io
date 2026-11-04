import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen factors
    # U2: 4 rows (00, 01, 10, 11), 7 cols
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # 00 (A11)
        [0, 0, 0, 0, 1, 0, 1],   # 01 (A12)
        [0, 1, 0, 0, 0, 1, 0],   # 10 (A21)
        [1, 1, 0, 1, 0, 0, -1]   # 11 (A22)
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],  # 00 (B11)
        [0, 0, 1, 0, 0, 1, 0],   # 01 (B12)
        [0, 0, 0, 1, 0, 0, 1],   # 10 (B21)
        [1, 0, -1, 0, 1, 0, 1]   # 11 (B22)
    ]
    
    # W2: 4 rows (C11, C21, C12, C22) -> Indices 0, 1, 2, 3
    # C11: 0, C21: 1, C12: 2, C22: 3
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [1, -1, 1, 0, 0, 1, 0]   # C22
    ]
    
    # Block level coefficients
    # alpha[k][u][v]
    # beta[k][u][v]
    # gamma[k][u][v]
    # k is 0..6, u,v is 0..1
    
    alpha = [
        [[1, 0], [0, 1]],  # k=0: A00 + A11
        [[0, 0], [1, 1]],  # k=1: A10 + A11
        [[1, 0], [0, 0]],  # k=2: A00
        [[0, 0], [0, 1]],  # k=3: A11
        [[1, 1], [0, 0]],  # k=4: A00 + A01
        [[-1, 0], [1, 0]], # k=5: A10 - A00
        [[0, 1], [0, -1]]  # k=6: A01 - A11
    ]
    
    beta = [
        [[1, 0], [0, 1]],  # k=0: B00 + B11
        [[1, 0], [0, 0]],  # k=1: B00
        [[0, 1], [0, -1]], # k=2: B01 - B11
        [[-1, 0], [1, 0]], # k=3: B10 - B00
        [[0, 0], [0, 1]],  # k=4: B11
        [[1, 1], [0, 0]],  # k=5: B00 + B01
        [[0, 0], [1, 1]]   # k=6: B10 + B11
    ]
    
    gamma = [
        [[1, 0], [0, 1]],  # k=0: C00 + C11
        [[0, 0], [1, -1]], # k=1: C10 - C11
        [[0, 1], [0, 1]],  # k=2: C01 + C11
        [[1, 0], [1, 0]],  # k=3: C00 + C10
        [[-1, 1], [0, 0]], # k=4: -C00 + C01
        [[0, 0], [0, 1]],  # k=5: C11
        [[1, 0], [0, 0]]   # k=6: C00
    ]
    
    for k in range(7):
        for s in range(7):
            col = 7 * k + s
            
            # Fill U
            for u in range(2):
                for v in range(2):
                    a_val = alpha[k][u][v]
                    if a_val != 0:
                        for r in range(2):
                            for c in range(2):
                                # Global A index: i = 2u + r, j = 2v + c
                                i = 2 * u + r
                                j = 2 * v + c
                                u_idx = i * m + j
                                # Local U2 index: row = r*2+c? No, U2 rows are 00,01,10,11.
                                # r is local row, c is local col.
                                # U2 index is r*2 + c.
                                u2_val = U2[r * 2 + c][s]
                                U[u_idx][col] += a_val * u2_val
                                
            # Fill V
            for u in range(2):
                for v in range(2):
                    b_val = beta[k][u][v]
                    if b_val != 0:
                        for r in range(2):
                            for c in range(2):
                                # Global B index: j = 2u + r, k_col = 2v + c
                                j = 2 * u + r
                                k_col = 2 * v + c
                                v_idx = j * p + k_col
                                v2_val = V2[r * 2 + c][s]
                                V[v_idx][col] += b_val * v2_val
                                
            # Fill W
            for u in range(2):
                for v in range(2):
                    g_val = gamma[k][u][v]
                    if g_val != 0:
                        for r in range(2):
                            for c in range(2):
                                # Global C index: i_row = 2u + r, k_col = 2v + c
                                i_row = 2 * u + r
                                k_col = 2 * v + c
                                # W index: k_col * n + i_row
                                w_idx = k_col * n + i_row
                                # Local W2 index: row = c*2 + r?
                                # W2 rows are C11, C21, C12, C22.
                                # C11: i=0, k=0 -> 0
                                # C21: i=1, k=0 -> 1
                                # C12: i=0, k=1 -> 2
                                # C22: i=1, k=1 -> 3
                                # So index is k_local * 2 + i_local.
                                # Here r is i_local, c is k_local.
                                w2_val = W2[c * 2 + r][s]
                                W[w_idx][col] += g_val * w2_val
                                
    return {"U": U, "V": V, "W": W}
