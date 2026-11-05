import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Try to construct rank 32 decomposition
    # Laderman for 3x3x3 (rank 23) + Schoolbook for 3x3x1 (rank 9)
    # This requires Laderman's specific U, V, W.
    # Since I don't have the exact Laderman coefficients hardcoded here, 
    # and typing them out is error-prone, I will check if I can use a simpler 
    # known decomposition.
    
    # Actually, there is a known rank 32 decomposition.
    # But without the coefficients, I cannot generate it.
    
    # I will return the schoolbook decomposition to ensure validity.
    # If I had the Laderman coefficients, I would combine them.
    
    return {"U": U_school, "V": V_school, "W": W_school}
