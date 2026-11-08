import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Try ranks from 29 down to 25
    for r in range(29, 24, -1):
        if time.time() > deadline:
            break
        if search_rank(r, n, m, p):
            return get_decomposition(r, n, m, p)
            
    # Fallback to schoolbook rank 30
    r = n * m * p
    return get_decomposition(r, n, m, p)
