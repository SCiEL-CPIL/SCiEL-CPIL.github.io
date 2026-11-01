import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to find rank 45 solution (9 * rank 5 for 2x3x1)
    # If successful, return it. Else return rank 54.
    
    # Rank 54 is always valid.
    def get_rank_54():
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

    # The rank 6 construction is mathematically invalid for this tensor size.
    # The naive rank is n*m*p = 54.
    # We return the valid naive decomposition.
    return get_rank_54()
