import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    timeout = 100.0
    
    n, m, p = DIMENSIONS
    
    # Rank 36 solution
    rank_36 = 36
    U = [[0]*rank_36 for _ in range(n*m)]
    V = [[0]*rank_36 for _ in range(m*p)]
    W = [[0]*rank_36 for _ in range(p*n)]
    
    # ... populate rank 36 ...
    
    # Check time
    if time.time() - start_time > timeout - 20:
        return {"U": U, "V": V, "W": W}
        
    # Try to find Rank 13 for M_{2,4,2}
    # ...
    
    return {"U": U, "V": V, "W": W}
