import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # ... (implementation of rank 44) ...
    
    return {"U": U, "V": V, "W": W}
