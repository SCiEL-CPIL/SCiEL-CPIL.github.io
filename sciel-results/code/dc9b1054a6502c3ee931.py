"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = 2, 3, 8
    
    # Rank-6 decomposition: T = sum_{i,j} e_{i*m+j} (x) v_j (x) w_i
    # where v_j = sum_k e_{j*p+k} and w_i = sum_k e_{k*n+i}
    rank = n * m  # 2*3 = 6
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            # U: e_{i*m+j}
            U[i * m + j][col] = 1
            # V: v_j = sum_k e_{j*p+k}
            for k in range(p):
                V[j * p + k][col] = 1
            # W: w_i = sum_k e_{k*n+i}
            for k in range(p):
                W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
