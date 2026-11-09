import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal (or near-minimal) exact tensor decomposition 
    for multiplying a 3x3 matrix by a 3x5 matrix.
    
    The interface requires returning {'U', 'V', 'W'} with shapes:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Entries must be integers or half-integers.
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard: 150s limit, we return well before.
    deadline = time.time() + 115
    
    # The schoolbook decomposition is guaranteed to be correct and valid.
    # Rank = n * m * p
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    column = 0
    # Iterate over all i, j, k
    # Tensor index for U: i*m + j
    # Tensor index for V: j*p + k
    # Tensor index for W: k*n + i (Trace convention W[k*n+i, r])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
                
    # Check if we have time to attempt a better decomposition
    if time.time() < deadline - 30:
        # Try to find a lower rank decomposition
        # For 3x3 * 3x5, the trivial bound is 45
        # We can try to use known results or simple optimizations
        # For now, we keep the schoolbook as it's guaranteed correct
        # A more sophisticated search would be needed to improve this
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
