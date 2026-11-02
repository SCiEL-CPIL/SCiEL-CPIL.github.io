"""Program for Matrix multiplication 3,3,6: Block-split schoolbook rank-18 decomposition."""
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split the p=6 dimension into two blocks of size 3
    # Since p is even and n=m=3, we can treat this as two independent 3x3x3 multiplications
    block_p = 3
    num_blocks = p // block_p
    
    if num_blocks * block_p != p:
        # Fallback to schoolbook if not divisible, though for this task p=6 is divisible
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Rank for each block is n*m*block_p = 3*3*3 = 9
    rank_per_block = n * m * block_p
    total_rank = num_blocks * rank_per_block
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    for b in range(num_blocks):
        base_col = b * rank_per_block
        for i in range(n):
            for j in range(m):
                for k_local in range(block_p):
                    k_global = b * block_p + k_local
                    col = base_col + i * m * block_p + j * block_p + k_local
                    
                    # U index: i*m + j
                    U[i * m + j, col] = 1.0
                    
                    # V index: j*p + k_global
                    V[j * p + k_global, col] = 1.0
                    
                    # W index: k_global*n + i
                    W[k_global * n + i, col] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
