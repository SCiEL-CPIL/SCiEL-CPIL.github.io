"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s deadline to be safe
    
    n, m, p = 2, 4, 4
    # Rank 8 decomposition using schoolbook method per row of A
    rank = 8
    
    U = [[0] * rank for _ in range(n * m)]  # (8, 8)
    V = [[0] * rank for _ in range(m * p)]  # (16, 8)
    W = [[0] * rank for _ in range(p * n)]  # (8, 8)
    
    # A is 2x4, B is 4x4, C is 2x4
    # C[i, j] = sum_k A[i, k] * B[k, j]
    # We decompose this into 8 rank-1 terms.
    # For each k in 0..3, and each i in 0..1, we have a term.
    # r = i * 4 + k
    
    for i in range(n):
        for k in range(m):
            r = i * m + k
            
            # U index: A[i, k] -> i * m + k
            u_idx = i * m + k
            U[u_idx][r] = 1
            
            # V indices: B[k, j] for all j in 0..p-1
            # V index: k * p + j
            for j in range(p):
                v_idx = k * p + j
                V[v_idx][r] = 1
                
            # W indices: C[i, j] for all j in 0..p-1
            # W index: j * n + i (trace convention)
            for j in range(p):
                w_idx = j * n + i
                W[w_idx][r] = 1

    if time.time() > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
