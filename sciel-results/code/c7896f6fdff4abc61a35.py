import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 using 4 blocks of Strassen rank 7
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Correct Strassen terms for 2x2x2
    # U: 2x2, V: 2x2, W: 2x2 (W[k][i])
    strassen_terms = [
        # 1: M1
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [1, 0]]),
        # 2: M2
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, -1]]),
        # 3: M3
        ([[1, 0], [0, 0]], [[0, 0], [1, -1]], [[0, 1], [0, 1]]),
        # 4: M4
        ([[0, 0], [0, 1]], [[-1, 1], [0, 0]], [[-1, 1], [0, 0]]),
        # 5: M5
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[1, 0], [0, 0]]),
        # 6: M6
        ([[-1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        # 7: M7
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[-1, 0], [0, -1]])
    ]
    
    # Map blocks
    # Block 0: i 0-1, j 0-1, k 0-1
    # Block 1: i 0-1, j 0-1, k 2-3
    # Block 2: i 0-1, j 2-3, k 0-1
    # Block 3: i 0-1, j 2-3, k 2-3
    
    col = 0
    for b in range(4):
        # Determine offsets
        i_off = 0
        j_off = (b // 2) * 2
        k_off = (b % 2) * 2
        
        for r in range(7):
            u, v, w = strassen_terms[r]
            
            # Populate U
            for i_loc in range(2):
                for j_loc in range(2):
                    val = u[i_loc][j_loc]
                    if val != 0:
                        i_g = i_loc + i_off
                        j_g = j_loc + j_off
                        U[i_g * m + j_g][col] = val
                        
            # Populate V
            for j_loc in range(2):
                for k_loc in range(2):
                    val = v[j_loc][k_loc]
                    if val != 0:
                        j_g = j_loc + j_off
                        k_g = k_loc + k_off
                        V[j_g * p + k_g][col] = val
                        
            # Populate W
            # Index is k*n + i
            # Value is w[k_loc][i_loc]
            for k_loc in range(2):
                for i_loc in range(2):
                    val = w[k_loc][i_loc]
                    if val != 0:
                        k_g = k_loc + k_off
                        i_g = i_loc + i_off
                        W[k_g * n + i_g][col] = val
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
