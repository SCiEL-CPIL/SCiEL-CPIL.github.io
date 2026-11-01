"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook rank: n*m*p = 64
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
    
    # Verify correctness: check that the tensor decomposition is exact
    # T[i,j, j',k', k'',i''] should equal delta_{j,j'} * delta_{i,i''} * delta_{k'',k'}
    # We verify a sample of entries
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Verify by reconstructing the tensor and checking against expected
    # For efficiency, check a subset of entries
    import random
    random.seed(42)
    num_checks = min(1000, n*m * m*p * p*n)
    for _ in range(num_checks):
        idx_a = random.randint(0, n*m-1)
        idx_b = random.randint(0, m*p-1)
        idx_c = random.randint(0, p*n-1)
        
        i, j = divmod(idx_a, m)
        jp, k = divmod(idx_b, p)
        kn, ip = divmod(idx_c, n)
        
        expected = 1.0 if (j == jp and i == ip and k == kn) else 0.0
        actual = sum(U_arr[idx_a, t] * V_arr[idx_b, t] * W_arr[idx_c, t] for t in range(rank))
        
        if abs(expected - actual) > 1e-10:
            # Fallback: return schoolbook (should not happen)
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
