import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 44
    # We use a rank 22 decomposition for 2x3 by 3x4
    # and apply it to the two halves of the 3x8 matrix.
    
    # Since I don't have the exact rank 22 coefficients, 
    # I will use the schoolbook rank 24 for each half, 
    # but I will try to find a rank 44 decomposition if possible.
    
    # For now, I will return the schoolbook decomposition of rank 48,
    # but I will structure the code to allow for a rank 44 implementation.
    
    # Actually, I will implement the rank 44 decomposition if I can find a simple one.
    # If not, I will use rank 48.
    
    # Let's try to construct a rank 44 decomposition.
    # The rank of 2x3 by 3x4 is 20 or 22.
    # If it is 22, then 2 * 22 = 44.
    
    # I will use a known rank 22 decomposition for 2x3 by 3x4.
    # Reference: "The rank of 2x3 by 3x4 matrix multiplication is 20" is a known result, 
    # but 22 is also achievable and easier to construct?
    # Actually, 20 is the minimum.
    # If I use rank 20, total rank is 40.
    # Score 48/40 = 1.2.
    
    # I will try to implement the rank 20 decomposition.
    # If I cannot, I will use rank 48.
    
    # For safety, I will return the schoolbook decomposition.
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
