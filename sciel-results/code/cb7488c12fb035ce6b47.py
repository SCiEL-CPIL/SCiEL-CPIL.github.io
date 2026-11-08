import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = _t0 + 145.0

    n, m, p = 6, 6, 6
    
    # Direct rank-216 identity decomposition (naive algorithm)
    # This serves as a baseline model distinct from Strassen-based approaches.
    # Rank = n * m * p = 216
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col_idx] = 1
                # V index: j*p + k
                V[j * p + k][col_idx] = 1
                # W index: k*n + i
                W[k * n + i][col_idx] = 1
                col_idx += 1
                
                if time.time() > _deadline:
                    break
            if time.time() > _deadline:
                break
        if time.time() > _deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
