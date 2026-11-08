import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s (and 150s) limit

    n, m, p = 3, 4, 5

    # Schoolbook decomposition: Rank = n * m * p
    # Each term (i, k, j) gets its own rank dimension.
    # U[(i*m + k), r] = 1
    # V[(k*p + j), r] = 1
    # W[(j*n + i), r] = 1
    # where r = i * (m * p) + k * p + j
    
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                r = i * (m * p) + k * p + j
                
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
