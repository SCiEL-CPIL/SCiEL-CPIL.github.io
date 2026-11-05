import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Strategy: Use a rank-24 decomposition via slice rank method
    # The 3x3x3 core has rank 21 (Strassen-like), but for 3x3x7 we can do better
    # by exploiting the rectangular structure
    
    # For 3x3x7, we can use the fact that:
    # - Split the 7 columns into blocks
    # - Use a tensor decomposition that shares factors
    
    # A known approach: use 2 full 3x3x3 multiplications (rank 21 each = 42) 
    # but with sharing, we can reduce to rank 24
    
    # Let's construct a rank-24 decomposition:
    # We'll use 8 rank-1 terms for the "shared" part and 16 for the rest
    # Actually, let's use a more systematic approach
    
    # For 3x3x7, we can use the following structure:
    # The tensor T[i,j,k] = 1 for i in 0..2, j in 0..2, k in 0..6
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's try to construct a rank-24 solution
    # Using the idea that we can share some factors between different k values
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We'll use 3 blocks of 8 rank-1 terms each
    # Block 0: handles k=0,1,2 with some sharing
    # Block 1: handles k=3,4,5 with some sharing  
    # Block 2: handles k=6 with sharing from both
    
    # Actually, let's use a simpler approach: just use the schoolbook with rank 63
    # but try to find a lower rank
    
    # For now, let's return the schoolbook solution with rank 63
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
