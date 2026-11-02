import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
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
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                # Check time
                if time.time() > deadline:
                    # If we run out of time, return what we have?
                    # But we need to complete the loop to have a valid decomposition.
                    # The loop is small (105 iterations), so this won't happen.
                    pass
                    
    return {"U": U, "V": V, "W": W}
