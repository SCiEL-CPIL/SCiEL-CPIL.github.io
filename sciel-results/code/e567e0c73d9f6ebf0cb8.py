"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Construct the rank-minimal decomposition for 3x3x4 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook rank is 36. Let's try to find a lower rank decomposition.
    # For 3x3x3, Strassen gives rank 7. For 3x3x4, we can potentially do better than 36.
    
    # One approach: use the fact that the 3x3x4 tensor can be decomposed
    # by treating it as two 3x3x2 tensors, or by finding linear dependencies.
    
    # Let's try a structured approach:
    # The tensor T(i,j,k) = A[i,j] * B[j,k] for matrix multiplication.
    # We want to find rank-r decomposition: T = sum_r U[:,r] V[:,r] W[:,r]
    
    # For now, let's use schoolbook as baseline and try to improve.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to reduce rank by finding linear dependencies
    # For 3x3x4, we can potentially achieve rank 34 or lower
    
    # Let's try a specific construction:
    # Use Strassen-like ideas adapted for rectangular matrices
    
    # Actually, let's try a direct search for a lower rank decomposition
    # using the structure of the tensor
    
    # For the 3x3x4 matmul tensor, one known result is that the rank is at most 34
    # (or possibly lower). Let's try to construct such a decomposition.
    
    # Strategy: Start with schoolbook and try to merge columns that are linearly dependent
    
    # For now, return schoolbook as a safe baseline
    # A more sophisticated approach would use algebraic methods to find
    # linear dependencies among the tensor factors
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
