import time
import sys

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Check if we have time (we do, but good practice)
    if time.time() > deadline:
        # Return a trivial or best-effort result if we somehow timed out (unlikely)
        # For this small problem, we just proceed.
        pass

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
                # Optional: check time in loop if we were doing heavy search
                # if time.time() > deadline: break
                # But here the loop is tiny.
                
    return {"U": U, "V": V, "W": W}
