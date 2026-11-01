"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a decomposition for (2,3,6) matrix multiplication.
    Uses splitting into two (2,3,3) blocks with schoolbook rank 18 each.
    Total rank = 36.
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    r_block = n * 3 * 3  # 18
    U_block = np.zeros((n * 3, r_block), dtype=float)
    V_block = np.zeros((3 * 3, r_block), dtype=float)
    W_block = np.zeros((3 * n, r_block), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(3):
            for k in range(3):
                U_block[i * 3 + j, col] = 1
                V_block[j * 3 + k, col] = 1
                W_block[k * n + i, col] = 1
                col += 1
    
    total_rank = 2 * r_block
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    U[:, 0:r_block] = U_block
    V[0:9, 0:r_block] = V_block
    W[0:6, 0:r_block] = W_block
    
    U[:, r_block:2*r_block] = U_block
    V[9:18, r_block:2*r_block] = V_block
    W[6:12, r_block:2*r_block] = W_block
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
