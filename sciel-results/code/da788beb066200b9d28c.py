import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Construct a tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    We use the schoolbook (rank 63) decomposition as a baseline, but we can do better.
    The key insight is that we can use Strassen's algorithm for the 3x3 core (rank 23)
    and extend it to handle the 7 columns.
    
    However, since implementing Strassen correctly for this specific case is complex,
    let's first try a hybrid approach:
    - For the first 3 columns of the output (k=0,1,2), use Strassen-like rank 23
    - For the remaining 4 columns (k=3,4,5,6), use schoolbook rank 9 each
    
    Total rank would be 23 + 4*9 = 23 + 36 = 59
    
    But wait, we need to be careful about how the factors interact.
    
    Actually, let's just implement the full schoolbook for now to ensure correctness,
    then we can optimize.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook rank 63
    rank = n * m * p
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # shape (9, rank)
    V = [[0] * rank for _ in range(m * p)]  # shape (21, rank)
    W = [[0] * rank for _ in range(p * n)]  # shape (21, rank)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
