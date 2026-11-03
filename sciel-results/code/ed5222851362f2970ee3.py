import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix.
    
    Uses schoolbook multiplication with rank 36.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank) = (9, rank)
        - V has shape (m*p, rank) = (12, rank)
        - W has shape (p*n, rank) = (12, rank)
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 3, 3, 4
    rank = n * m * p  # 36
    
    U = [[0] * rank for _ in range(n * m)]   # (9, 36)
    V = [[0] * rank for _ in range(m * p)]   # (12, 36)
    W = [[0] * rank for _ in range(p * n)]   # (12, 36)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
