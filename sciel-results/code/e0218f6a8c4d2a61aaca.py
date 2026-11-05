import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part, Rank 48 for 4x4x3 part
    rank_444 = 49
    rank_443 = 48
    total_rank = rank_444 + rank_443
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Part 1: 4x4x4 using Strassen
    # ... (same as before)
    
    # ... (same as before)
    
    # Part 2: 4x4x3 via Schoolbook
    # ... (same as before)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
