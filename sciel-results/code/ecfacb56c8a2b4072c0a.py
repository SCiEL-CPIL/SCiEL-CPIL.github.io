import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    t_start = time.time()
    DEADLINE = 145
    n, m, p = DIMENSIONS  # 3, 3, 7

    def schoolbook():
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

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def time_left():
        return DEADLINE - (time.time() - t_start)

    # Fallback: schoolbook
    result = schoolbook()
    if not verify(result['U'], result['V'], result['W']):
        raise RuntimeError("Schoolbook verification failed!")
    
    return result
