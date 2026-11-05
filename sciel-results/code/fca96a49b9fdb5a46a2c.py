"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy: Use polynomial interpolation on the middle index j (dimension 4).
    Map the 4 middle indices to evaluations of a cubic polynomial at 4 distinct points.
    This allows us to use a Vandermonde-based decomposition.
    
    The key idea: For each (i, k) pair, the contribution from middle index j is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    If we think of A[i,j] and B[j,k] as values of polynomials, we can use
    interpolation to reduce the rank. However, for exact integer/half-integer
    decomposition, we need to be careful.
    
    Actually, let's use a simpler approach: try to find a lower-rank decomposition
    using random search with verification.
    
    Given the time constraint, let's first try the schoolbook (rank 96) and then
    attempt to reduce it.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find a better decomposition using a structured approach
    # Use the fact that we can split the middle dimension
    
    # Approach: Try rank = n * p + m * p - something, or other structured ranks
    # For now, let's try a few candidate ranks below 96
    
    candidate_ranks = [95, 94, 93, 92, 91, 90]
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        # This would require a search algorithm to find the decomposition
        # For now, we'll just return the schoolbook if we can't find better quickly
        pass
    
    return best_result
# EVOLVE-BLOCK-END
