"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen parameters for 2x2
    # u: list of (i, j, val)
    # v: list of (j, k, val)
    # w: list of (k, i, val)
    
    strassen_u = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]
    
    strassen_v = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    
    strassen_w = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    # 5 blocks of 2x2
    for t in range(5):
        for r in range(7):
            # U
            for i, j, val in strassen_u[r]:
                U[i * m + j][col] = val
            
            # V
            for j, k, val in strassen_v[r]:
                global_k = 2 * t + k
                V[j * p + global_k][col] = val
            
            # W
            for k, i, val in strassen_w[r]:
                global_k = 2 * t + k
                W[global_k * n + i][col] = val
            
            col += 1
    
    # 1 block of 2x1 (column 10)
    # Terms:
    # 1. A00 * B0,10 -> C0,10
    U[0 * m + 0][col] = 1
    V[0 * p + 10][col] = 1
    W[10 * n + 0][col] = 1
    col += 1
    
    # 2. A01 * B1,10 -> C0,10
    U[0 * m + 1][col] = 1
    V[1 * p + 10][col] = 1
    W[10 * n + 0][col] = 1
    col += 1
    
    # 3. A10 * B0,10 -> C1,10
    U[1 * m + 0][col] = 1
    V[0 * p + 10][col] = 1
    W[10 * n + 1][col] = 1
    col += 1
    
    # 4. A11 * B1,10 -> C1,10
    U[1 * m + 1][col] = 1
    V[1 * p + 10][col] = 1
    W[10 * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
