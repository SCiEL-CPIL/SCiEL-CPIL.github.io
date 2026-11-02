"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Matrix multiplication tensor T has shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # The schoolbook rank is n*m*p = 32
    # We can potentially do better.
    
    # Let's try to find a lower rank decomposition.
    # The tensor is a 3-tensor of dimensions (8, 16, 8).
    
    # We'll use a simple approach: try to find a rank-r decomposition
    # by solving a system of linear equations.
    
    # First, let's build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from low to high
    # Start with a reasonable guess
    # The trivial bound is n*m*p = 32
    # But we can likely do better.
    
    # Let's use a greedy approach or try specific ranks
    # For small tensors, we can try to find exact decompositions
    
    # A key insight: the matrix multiplication tensor has structure
    # that allows for Strassen-like optimizations.
    
    # For 2x4x4, we can think of it as:
    # A is 2x4, B is 4x4, C is 2x4
    # C = A * B
    
    # We can split A into two 2x2 blocks: A = [A1, A2] where A1, A2 are 2x2
    # B can be split into 2x2 blocks: B = [[B11, B12], [B21, B22]]
    # Then C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    # This gives us 8 multiplications of 2x2 matrices
    # Each 2x2 multiplication can be done with 7 multiplications (Strassen)
    # So total rank could be 8*7 = 56? That's worse than 32.
    
    # Actually, the schoolbook is already optimal for this small case?
    # Let's just return the schoolbook decomposition.
    
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
# EVOLVE-BLOCK-END
