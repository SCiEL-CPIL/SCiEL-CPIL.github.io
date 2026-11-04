import time

def construct():
    """
    Returns a rank-144 schoolbook tensor decomposition for 4x4 * 4x9 matrix multiplication.
    U: (4*4, 144)
    V: (4*9, 144)
    W: (9*4, 144)
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 4, 4, 9
    rank = n * m * p  # 144
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # If we hit the deadline, trim to consistent rank
    if col < rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
