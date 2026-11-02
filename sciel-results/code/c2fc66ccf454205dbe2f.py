import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    n, m, p = DIMENSIONS
    
    # Rank 42 decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # x, y, z are 2x2 lists
    strassen_steps = [
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]), # M1
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]), # M2
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 1]]), # M3
        ([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 0], [1, 0]]), # M4
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 0], [0, 1]]), # M5
        ([[-1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]), # M6
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]])  # M7
    ]
    
    # Blocks: (ai, aj, bj, bk, ci, ck)
    # ai, aj: offset in A (2x4)
    # bj, bk: offset in B (4x6)
    # ci, ck: offset in C (2x6)
    blocks = [
        (0, 0, 0, 0, 0, 0), # A11 B11 -> C11
        (0, 2, 2, 0, 0, 0), # A12 B21 -> C11
        (0, 0, 0, 2, 0, 2), # A11 B12 -> C12
        (0, 2, 2, 2, 0, 2), # A12 B22 -> C12
        (0, 0, 0, 4, 0, 4), # A11 B13 -> C13
        (0, 2, 2, 4, 0, 4)  # A12 B23 -> C13
    ]
    
    col = 0
    for (ai, aj, bj, bk, ci, ck) in blocks:
        for (x, y, z) in strassen_steps:
            for u in range(2):
                for v in range(2):
                    if x[u][v] != 0:
                        U[(ai + u) * m + (aj + v)][col] = x[u][v]
                    if y[u][v] != 0:
                        V[(bj + u) * p + (bk + v)][col] = y[u][v]
                    if z[u][v] != 0:
                        W[(ck + v) * n + (ci + u)][col] = z[u][v]
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, W: W}
