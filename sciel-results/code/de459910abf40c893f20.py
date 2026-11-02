import time

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
_TIME_LIMIT = 110.0

def construct():
    # Check deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to schoolbook (fast path)
        pass
        
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
