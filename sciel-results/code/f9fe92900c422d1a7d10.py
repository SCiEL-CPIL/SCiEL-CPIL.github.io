"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will attempt to find a lower rank decomposition.
    # The schoolbook rank is n*m*p = 3*3*7 = 63.
    # We can potentially use Strassen-like decompositions for blocks.
    # A 3x3 matrix multiplication can be seen as:
    # C = A * B
    # We can partition into 3x3 blocks if we pad or split.
    # Actually, let's try to construct a rank 55 decomposition by using
    # Strassen for the 3x3 part of the 7 columns? No, 7 is not a power of 2.
    #
    # Let's look at the structure.
    # A is 3x3, B is 3x7.
    # C is 3x7.
    # We can compute C[:, 0:3] and C[:, 3:7] separately?
    # C[:, 0:3] is 3x3.
    # C[:, 3:7] is 3x4.
    #
    # Rank(3x3) <= 23 (Strassen 3x3 decomposition)
    # Rank(3x4) ?
    #
    # Let's try a simpler approach: random search for rank < 63.
    # However, finding a valid tensor decomposition via random search is hard.
    
    # Fallback to schoolbook to ensure correctness and non-zero score.
    # If we can't find a better one quickly, return schoolbook.
    
    deadline = time.time() + 140.0
    best_rank = n * m * p
    best_decomp = None
    
    # Try to generate a random low-rank decomposition?
    # This is generally invalid unless the tensor lies in that subspace.
    # The matrix multiplication tensor is highly structured.
    # Random search is unlikely to find the exact coefficients without a specific model.
    
    # Given the strict time limit and the difficulty of finding Strassen coefficients for 3x3x7 from scratch,
    # and the risk of invalidity, I will stick to the schoolbook decomposition for now,
    # unless I can implement a known 3x3 Strassen decomposition.
    
    # Standard Strassen for 2x2 is rank 7.
    # For 3x3, one can partition into:
    # A = [A11 A12; A21 A22] where A11, A22 are 2x2? No, 3 is not even.
    # A = [A1 A2; A3] ... this is messy.
    
    # There is a known rank 23 decomposition for 3x3x3.
    # Let's see if we can hardcode it or derive it.
    # Since I don't have the coefficients memorized, and generating them is complex,
    # I will return the schoolbook decomposition.
    # This guarantees a valid result with rank 63 (score 1.0).
    # To improve, one would need to implement the 3x3 Strassen coefficients.
    
    rank = best_rank
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
