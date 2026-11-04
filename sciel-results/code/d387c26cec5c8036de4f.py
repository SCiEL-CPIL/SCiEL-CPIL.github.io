<<<<<<< SEARCH
def construct():
    global _start_time, _decompose_cache
    _start_time = time.time()
    _decompose_cache = {}
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    U, V, W = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
=======
def construct():
    global _start_time, _decompose_cache
    _start_time = time.time()
    _decompose_cache = {}
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    try:
        U, V, W = _decompose(n, m, p)
    except TimeoutError:
        # Fallback to schoolbook if we run out of time
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
>>>>>>> REPLACE

Now let me also reduce the TIME_LIMIT to give more safety margin, and add a check in the main loop to bail out early if we're close to the limit:

