"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    Strategy:
    1. Embed the 4x5x5 multiplication into a 5x5x5 multiplication by padding the 4x5 matrix to 5x5.
    2. Use the known rank-50 decomposition for 5x5 matrix multiplication (via Strassen-based or known bounds).
    3. Project back to the 4x4 output space.
    
    However, since we need an EXACT decomposition with integer/half-integer entries, and the known 
    rank for 5x5 matrix multiplication is not trivially 50 with simple structure, let's consider:
    
    Actually, the schoolbook rank is 4*5*5 = 100.
    
    Let's think about this more carefully. The task asks for rank-minimal exact decomposition.
    
    For matrix multiplication of n x m by m x p, the tensor is T in R^{(n*m) x (m*p) x (p*n)}.
    
    One approach: Use the fact that 4x5x5 can be embedded in 5x5x5. The rank of 5x5 matrix multiplication 
    is known to be at most 49 (via Strassen-like algorithms), but constructing an explicit decomposition 
    with integer/half-integer entries is non-trivial.
    
    Alternative: Use the direct construction with a lower rank than 100.
    
    Let me reconsider. The simplest correct approach that guarantees a valid answer is the schoolbook 
    decomposition of rank 100. But we can do better.
    
    Key insight: We can use the standard Strassen algorithm adapted for rectangular matrices.
    
    For 4x5 * 5x5:
    - Pad A (4x5) to 5x5 by adding a zero row: A' = [[A], [0,0,0,0,0]]
    - Compute C' = A' * B (5x5 * 5x5 = 5x5) using a rank-49 or lower decomposition
    - Take the top 4 rows of C' as the result C (4x5)
    
    The rank of 5x5 matrix multiplication: The best known upper bound is 49, but an explicit 
    decomposition with simple coefficients is complex.
    
    Let's use a simpler approach: Since we need integer/half-integer entries, let's try to find 
    a decomposition with rank < 100 using a structured approach.
    
    Actually, let's just return the schoolbook decomposition of rank 100 as a guaranteed valid 
    answer, since constructing a lower-rank decomposition with the required properties (integer/
    half-integer entries, exactness) is complex and time-consuming.
    
    Wait, the score is the rank, and lower is better. The schoolbook gives 100. Can we do better?
    
    Let me try a different approach: Use the fact that matrix multiplication is bilinear and 
    try to find a lower-rank decomposition using known results.
    
    For now, let's return the schoolbook decomposition to ensure correctness, and note that 
    further optimization could reduce the rank.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # Schoolbook rank
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
