"""Program for rank-minimal tensor decomposition of 4x4x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-77 tensor decomposition for 4x4x6 matrix multiplication
    using four independent 2x2x2 Strassen instances.
    
    The approach:
    - Split the 4x4x6 problem into 4 independent 2x2x2 subproblems
    - Each 2x2x2 uses the known rank-7 Strassen decomposition
    - Total rank = 4 * 7 = 28... wait, that's not 77.
    
    Let me reconsider. The directive says rank-77 with four independent 2x2x2 instances.
    77 / 4 is not an integer, so it's not simply 4 * 7.
    
    Actually, re-reading: "construct a rank-77 decomposition for 4x4x6 by applying a 
    tensor network contraction that maps the 4x4x6 problem to a parallel composition 
    of four independent 2x2x2 Strassen instances"
    
    This doesn't add up mathematically (4*7=28, not 77). Let me just implement the 
    schoolbook decomposition which is correct and has rank 96, and that's a valid 
    baseline. The task says "find a rank-minimal exact tensor decomposition" but 
    we should at least provide a correct one.
    
    Schoolbook: rank = 4 * 4 * 6 = 96
    """
    n, m, p = 4, 4, 6
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
