import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    timeout = 120.0
    
    # Hypothesis: Use a rank-n*m*p decomposition where each term isolates a specific (i, j, k) tuple.
    # This guarantees an exact decomposition for any p.
    rank = n * m * p  # 4 * 2 * 12 = 48
    
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 48)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 48)
    W = [[0] * rank for _ in range(p * n)]  # shape (24, 48)
    
    for r in range(rank):
        if time.time() - start_time > timeout - 5:
            break
            
        # Decode r into i, j, k
        # r = i * (m * p) + j * p + k
        k = r % p
        j = (r // p) % m
        i = r // (m * p)
        
        # U_{i, j} = 1
        U[i * m + j][r] = 1
        
        # V_{j, k} = 1
        V[j * p + k][r] = 1
        
        # W_{k, i} = 1
        W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
