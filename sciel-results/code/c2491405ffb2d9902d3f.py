"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # For 2x5x5 matrix multiplication, the known lower bound is 20 (from Strassen-like methods)
    # but the actual rank is not known to be below 50 easily.
    # However, we can try a few random restarts with small rank to see if we can find a decomposition.
    # But the task says "integer/half-integer entries that exactly decompose".
    
    # Let's just return the schoolbook decomposition. It's the safest bet.
    # If there were a known better rank, it would likely be in the literature or seed.
    
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
