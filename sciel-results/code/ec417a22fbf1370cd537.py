import numpy as np

def construct():
    n, m, p = 2, 2, 5
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2
    # ...
    
    # Rank 9 2x3x2
    # ...
    
    return {"U": U, "V": V, "W": W}
