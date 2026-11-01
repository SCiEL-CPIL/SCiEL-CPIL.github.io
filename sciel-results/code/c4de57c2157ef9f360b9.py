"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Return a tensor decomposition for 2x2x5 matrix multiplication.
    
    The schoolbook rank is n*m*p = 2*2*5 = 20.
    We attempt to find a lower-rank exact decomposition using integer/half-integer entries.
    
    Known results: The tensor rank of 2x2x5 is known to be 14 (Landsberg).
    We hardcode a known rank-14 decomposition if available, or use a heuristic search.
    
    For now, we return the schoolbook decomposition (rank 20) as a valid baseline.
    In a more advanced implementation, we could embed the known rank-14 decomposition.
    """
    n, m, p = 2, 2, 5
    rank = n * m * p  # 20
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x rank
    V = [[0] * rank for _ in range(m * p)]  # 10 x rank
    W = [[0] * rank for _ in range(p * n)]  # 10 x rank
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
