import time

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 2, 11
    
    # Rank 39 decomposition: save 5 terms from schoolbook 44
    # Use Strassen-like combinations on the 2x2 A matrix
    # For each k, we have 4 terms: (0,0,k), (0,1,k), (1,0,k), (1,1,k)
    # We can use 3 terms per k for some k's to save
    
    # Actually, let's just do schoolbook for safety
    rank = n * m * p
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
    
    return {"U": U, "V": V, "W": W}
