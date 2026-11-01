import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen M_2,2,2
    # U_S is 4x7, V_S is 4x7, W_S is 4x7
    U_S = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V_S = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W_S = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Schoolbook M_1,2,3
    # U_S2 is 2x6, V_S2 is 6x6, W_S2 is 3x6
    U_S2 = np.array([
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1]
    ], dtype=int)
    
    V_S2 = np.eye(6, dtype=int)
    
    W_S2 = np.array([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1]
    ], dtype=int)
    
    rank = 7 * 6
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r in range(rank):
        r_S = r // 6
        r_S2 = r % 6
        
        # Fill U
        for row in range(n * m):
            i1 = row // 4
            rem = row % 4
            j1 = rem // 2
            j2 = rem % 2
            
            row_S = 2 * i1 + j1
            row_S2 = j2
            
            U[row, r] = U_S[row_S, r_S] * U_S2[row_S2, r_S2]
            
        # Fill V
        for row in range(m * p):
            j1 = row // 12
            rem1 = row % 12
            j2 = rem1 // 6
            rem2 = rem1 % 6
            k1 = rem2 // 3
            k2 = rem2 % 3
            
            row_S = 2 * j1 + k1
            row_S2 = 3 * j2 + k2
            
            V[row, r] = V_S[row_S, r_S] * V_S2[row_S2, r_S2]
            
        # Fill W
        for row in range(p * n):
            k1 = row // 6
            rem1 = row % 6
            k2 = rem1 // 2
            i1 = rem1 % 2
            
            row_S = 2 * k1 + i1
            row_S2 = k2
            
            W[row, r] = W_S[row_S, r_S] * W_S2[row_S2, r_S2]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
