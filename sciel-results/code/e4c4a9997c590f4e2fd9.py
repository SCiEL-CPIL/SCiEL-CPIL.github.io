"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Direct rank-216 identity decomposition (naive algorithm)
    # Rank = n * m * p
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
                # W index: k*n + i (trace convention)
                W[k * n + i][col_idx] = 1
                col_idx += 1
                
                if time.time() > deadline:
                    # Return partial result if time runs out, though this is very fast
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
