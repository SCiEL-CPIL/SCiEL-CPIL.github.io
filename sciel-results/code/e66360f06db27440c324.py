import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Rank 42 decomposition using 6 Strassen blocks
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42

    # Strassen coefficients for 2x2x2
    # U (Matrix A)
    U_strassen = [
        [(0, 1), (3, 1)],       # P1: A00 + A11
        [(2, 1), (3, 1)],       # P2: A10 + A11
        [(0, 1)],               # P3: A00
        [(3, 1)],               # P4: A11
        [(0, 1), (1, 1)],       # P5: A00 + A01
        [(2, 1), (0, -1)],      # P6: A10 - A00
        [(1, 1), (3, -1)]       # P7: A01 - A11
    ]
    
    # V (Matrix B)
    V_strassen = [
        [(0, 1), (3, 1)],       # P1: B00 + B11
        [(0, 1)],               # P2: B00
        [(1, 1), (3, -1)],      # P3: B01 - B11
        [(2, 1), (0, -1)],      # P4: B10 - B00
        [(3, 1)],               # P5: B11
        [(0, 1), (1, 1)],       # P6: B00 + B01
        [(2, 1), (3, 1)]        # P7: B10 + B11
    ]
    
    # W (Matrix C)
    W_strassen = [
        [(0, 1), (3, 1)],       # P1: C00 + C11
        [(2, 1), (3, -1)],      # P2: C10 - C11
        [(1, 1), (3, 1)],       # P3: C01 + C11
        [(0, 1), (2, 1)],       # P4: C00 + C10
        [(0, -1), (1, 1)],      # P5: -C00 + C01
        [(3, 1)],               # P6: C11
        [(0, 1)]                # P7: C00
    ]

    col = 0
    # 3 blocks of B columns (b=0,1,2)
    # 2 blocks of A columns (a=0,1)
    for b in range(3):
        for a in range(2):
            for s in range(7):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                    
                # Update U
                for (idx, coeff) in U_strassen[s]:
                    i_loc = idx // 2
                    j_loc = idx % 2
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V
                for (idx, coeff) in V_strassen[s]:
                    j_loc = idx // 2
                    k_loc = idx % 2
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W
                for (idx, coeff) in W_strassen[s]:
                    i_loc = idx // 2
                    k_loc = idx % 2
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}
