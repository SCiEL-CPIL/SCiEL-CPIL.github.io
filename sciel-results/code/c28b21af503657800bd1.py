import time
import sys

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Split m into 2 blocks of size 2
    # Split p into 3 blocks of size 2
    # Each block is 2x2x2, rank 7 (Strassen)
    # Total rank 2 * 3 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # S_t: 2x2 matrix for A
    # P_t: 2x2 matrix for B
    # Q_t: 2x2 matrix for C (index i, k)
    
    S = [
        [[1, 0], [0, 1]], # 1
        [[0, 0], [1, 1]], # 2
        [[1, 0], [0, 0]], # 3
        [[0, 0], [0, 1]], # 4
        [[1, 0], [1, 0]], # 5
        [[-1, 0], [1, 0]],# 6
        [[0, 1], [0, -1]] # 7
    ]
    
    P = [
        [[1, 0], [0, 1]], # 1
        [[1, 0], [0, 0]], # 2
        [[0, 0], [1, -1]],# 3
        [[-1, 1], [0, 0]],# 4
        [[0, 0], [0, 1]], # 5
        [[1, 1], [0, 0]], # 6
        [[0, 0], [1, 1]]  # 7
    ]
    
    # Q_t: C_{i,k} coefficient
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    Q = [
        [[1, 0], [0, 1]], # 1: C00, C11
        [[0, 0], [1, 0]], # 2: C10
        [[0, 1], [0, 0]], # 3: C01
        [[1, 0], [1, 0]], # 4: C00, C10
        [[-1, 1], [0, 0]],# 5: -C00, C01
        [[0, 0], [0, 1]], # 6: C11
        [[1, 0], [0, 0]]  # 7: C00
    ]
    
    col = 0
    # j_block in 0, 1
    # k_block in 0, 1, 2
    for j_block in range(2):
        for k_block in range(3):
            for t in range(7):
                # Local indices
                # i in 0, 1
                # j_local in 0, 1
                # k_local in 0, 1
                
                for i in range(2):
                    for j_local in range(2):
                        # Global j
                        j_global = j_block * 2 + j_local
                        U[i * m + j_global][col] = S[t][i][j_local]
                
                for j_local in range(2):
                    for k_local in range(2):
                        # Global j, k
                        j_global = j_block * 2 + j_local
                        k_global = k_block * 2 + k_local
                        V[j_global * p + k_global][col] = P[t][j_local][k_local]
                
                for k_local in range(2):
                    for i in range(2):
                        # Global k, i
                        # W index: k * n + i
                        k_global = k_block * 2 + k_local
                        W[k_global * n + i][col] = Q[t][i][k_local]
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
