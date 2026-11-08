"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * 7 = 49
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2
    # U, V, W are 2x2 matrices
    U2 = [
        [[1, 0], [0, 1]], # 1
        [[0, 0], [1, 1]], # 2
        [[1, 0], [0, 0]], # 3
        [[0, 0], [0, 1]], # 4
        [[1, 1], [0, 0]], # 5
        [[-1, 0], [1, 0]],# 6
        [[0, 1], [0, -1]] # 7
    ]
    V2 = [
        [[1, 0], [0, 1]], # 1
        [[1, 0], [0, 0]], # 2
        [[0, 1], [0, -1]],# 3
        [[0, 0], [1, -1]],# 4
        [[0, 0], [0, 1]], # 5
        [[1, 1], [0, 0]], # 6
        [[0, 0], [1, 1]]  # 7
    ]
    W2 = [
        [[1, 0], [0, 1]], # 1
        [[0, 1], [0, 0]], # 2
        [[0, 0], [1, 0]], # 3
        [[1, 0], [0, 0]], # 4
        [[0, 0], [0, 1]], # 5
        [[0, 0], [0, 1]], # 6
        [[0, 1], [0, 0]]  # 7
    ]
    
    col = 0
    for s in range(7):
        for t in range(7):
            # Kronecker product
            # U_kron = U2[s] x U2[t]
            # Shape 4x4
            U_kron = np.kron(np.array(U2[s]), np.array(U2[t]))
            V_kron = np.kron(np.array(V2[s]), np.array(V2[t]))
            W_kron = np.kron(np.array(W2[s]), np.array(W2[t]))
            
            # Flatten and fill
            for i in range(4):
                for j in range(4):
                    U[i * 4 + j][col] = U_kron[i, j]
                    V[i * 4 + j][col] = V_kron[i, j]
                    W[i * 4 + j][col] = W_kron[i, j]
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
