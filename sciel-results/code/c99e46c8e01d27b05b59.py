import time

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin within 120s limit
    
    n, m, p = DIMENSIONS
    rank = 9
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(rank):
        if time.time() > deadline:
            break
        j = t // p
        k = t % p
        
        V[j * p + k][t] = 1
        
        for i in range(n):
            U[i * m + j][t] = 1
            W[k * n + i][t] = 1
            
    return {"U": U, "V": V, "W": W}
