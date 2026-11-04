"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p  # 40
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for 120s limit
    
    # Try to find a lower rank decomposition
    # First, let's try to construct a rank-36 solution using Strassen's algorithm
    # for the 2x2 sub-blocks.
    
    # The 2x4 matrix can be split into two 2x2 blocks
    # The 4x5 matrix can be split into two 2x5 blocks (or one 2x4 and one 2x1)
    # Let's use Strassen for the 2x2 * 2x2 part and schoolbook for the rest.
    
    # Actually, let me think more carefully. The schoolbook rank is 40.
    # We can potentially reduce this by using Strassen's algorithm for 2x2 blocks.
    
    # Let's try to construct a rank-36 solution:
    # Split A (2x4) into A1 (2x2) and A2 (2x2)
    # Split B (4x5) into B1 (2x5) and B2 (2x5)
    # Split C (2x5) into C1 (2x5) and C2 (2x5)
    
    # For each 2x2 * 2x5 product, we can use a modified Strassen approach.
    # Actually, let me just try to find a valid decomposition with rank < 40.
    
    # Let's use a simpler approach: try to find a rank-39 solution by 
    # combining two schoolbook terms.
    
    # For now, let's return the schoolbook decomposition to ensure validity.
    # The rank-40 schoolbook is guaranteed to work.
    
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
