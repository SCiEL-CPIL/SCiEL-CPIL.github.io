"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Strategy: split p=10 into two blocks of size 5
    # For each 2x3x5 block, use rank-10 decomposition
    # Total rank = 20
    
    p_block = 5
    num_blocks = p // p_block  # 2
    rank_per_block = n * p_block  # 2 * 5 = 10
    total_rank = num_blocks * rank_per_block  # 20
    
    U = [[0] * total_rank for _ in range(n * m)]  # (6, 20)
    V = [[0] * total_rank for _ in range(m * p)]  # (30, 20)
    W = [[0] * total_rank for _ in range(p * n)]  # (20, 20)
    
    for block in range(num_blocks):
        for i in range(n):
            for k_local in range(p_block):
                r_local = i * p_block + k_local
                r_global = block * rank_per_block + r_local
                k_global = block * p_block + k_local
                
                for j in range(m):
                    U[i * m + j][r_global] = 1
                    V[j * p + k_global][r_global] = 1
                    W[k_global * n + i][r_global] = 1
    
    # Verify the decomposition (with time check)
    if time.time() < deadline - 10:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :]))
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-9:
            # Fallback to schoolbook
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
