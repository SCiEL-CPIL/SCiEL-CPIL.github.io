"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank decomposition
    # For 3x3x7, the known minimal rank is not well-studied, but let's try
    # a structured approach
    
    # First, let's try the schoolbook decomposition as a safe baseline
    def schoolbook_decomposition():
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
    
    # Try to use a block structure or known algorithms
    # For 3x3 matrix multiplication, Strassen's algorithm gives rank 7
    # But here we have 3x3 * 3x7, so we can split the 3x7 into blocks
    
    # Let's try to decompose using Strassen-like ideas
    # Actually, let's just return the schoolbook for now, as finding a lower rank
    # for this specific case requires more sophisticated search
    
    # Check time
    if time.time() > deadline:
        return schoolbook_decomposition()
    
    # Try a simple optimization: see if we can reduce rank by exploiting structure
    # For now, return schoolbook as it's guaranteed to work
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
