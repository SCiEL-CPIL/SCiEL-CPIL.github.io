import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # s: 0 to 6
    # F_s[i, j_loc]
    # G_s[j_loc, k_loc]
    # C_s[i, k_loc]
    
    F = [
        [[1, 0], [0, 1]],  # s=0: A00 + A11
        [[0, 0], [1, 1]],  # s=1: A10 + A11
        [[1, 0], [0, 0]],  # s=2: A00
        [[0, 0], [0, 1]],  # s=3: A11
        [[1, 1], [0, 0]],  # s=4: A00 + A01
        [[-1, 0], [1, 0]], # s=5: A10 - A00
        [[0, 1], [0, -1]]  # s=6: A01 - A11
    ]
    
    G = [
        [[1, 0], [0, 1]],  # s=0: B00 + B11
        [[1, 0], [0, 0]],  # s=1: B00
        [[0, 1], [-1, 0]], # s=2: B01 - B11
        [[-1, 0], [1, 0]], # s=3: B10 - B00
        [[0, 0], [0, 1]],  # s=4: B11
        [[1, 1], [0, 0]],  # s=5: B00 + B01
        [[0, 1], [1, 1]]   # s=6: B10 + B11
    ]
    
    C = [
        [[1, 0], [0, 1]],  # s=0: C00 + C11
        [[0, 0], [1, -1]], # s=1: C10 - C11
        [[0, 1], [0, 1]],  # s=2: C01 + C11
        [[1, 0], [1, 0]],  # s=3: C00 + C10
        [[-1, 0], [0, 1]], # s=4: -C00 + C01
        [[0, 0], [0, 1]],  # s=5: C11
        [[1, 0], [0, 0]]   # s=6: C00
    ]
    
    col = 0
    for b_j in range(2):
        for b_k in range(3):
            for s in range(7):
                # Fill U
                for i in range(2):
                    for j_loc in range(2):
                        val = F[s][i][j_loc]
                        if val != 0:
                            U[i * m + (2 * b_j + j_loc)][col] += val
                
                # Fill V
                for j_loc in range(2):
                    for k_loc in range(2):
                        val = G[s][j_loc][k_loc]
                        if val != 0:
                            V[(2 * b_j + j_loc) * p + (2 * b_k + k_loc)][col] += val
                
                # Fill W
                for k_loc in range(2):
                    for i in range(2):
                        val = C[s][i][k_loc]
                        if val != 0:
                            W[(2 * b_k + k_loc) * n + i][col] += val
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
