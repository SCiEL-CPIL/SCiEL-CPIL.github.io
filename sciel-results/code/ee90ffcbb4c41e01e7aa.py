import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time to ensure we don't exceed limit, though this is very fast
                if time.time() > deadline:
                    # In this specific case, the loop is tiny, so this won't trigger,
                    # but it satisfies the directive for a "deadline guard" structure.
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                # W follows trace convention W[k*n + i, r]
                W[k * n + i][column] = 1
                column += 1
        
        if time.time() > deadline:
            break
    if time.time() > deadline:
        break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
