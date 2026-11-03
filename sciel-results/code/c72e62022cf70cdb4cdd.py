"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy: Subspace embedding and null-space projection.
    1. Construct the schoolbook decomposition for the larger 4x4 * 4x8 tensor (n=4, m=4, p=8).
       This has rank 4*4*8 = 128.
    2. The 3x4 * 4x8 problem corresponds to restricting the first matrix A to have 3 rows.
       We can view this as embedding the 3x4 matrix into the top-left of a 4x4 matrix (zero-padding the 4th row).
    3. In the tensor decomposition of the 4x4 * 4x8 product, terms corresponding to the 4th row of A
       (i=3) involve U vectors with non-zero entries only in the indices corresponding to row 3.
       When projecting down to the 3x4 case (where U has dimension 3*4=12), these U vectors become zero.
       A term with a zero U vector contributes nothing to the tensor sum.
    4. Therefore, we can filter out all terms from the 4x4 * 4x8 schoolbook decomposition where i=3.
       The remaining terms correspond exactly to i in {0, 1, 2}, which gives a valid decomposition
       for the 3x4 * 4x8 tensor.
    5. The number of remaining terms is 3*4*8 = 96.
    """
    n_target, m, p = DIMENSIONS
    n_source = 4  # The larger dimension for the embedding
    
    # 1. Generate schoolbook terms for n_source x m x p
    # Each term is indexed by (i, j, k) with i in [0, n_source), j in [0, m), k in [0, p)
    terms = []
    for i in range(n_source):
        for j in range(m):
            for k in range(p):
                terms.append((i, j, k))
    
    # 2. Filter terms: keep only those where i < n_target
    # Terms with i >= n_target are in the "null space" of the projection to n_target rows.
    filtered_terms = [t for t in terms if t[0] < n_target]
    
    rank = len(filtered_terms)
    
    # 3. Construct U, V, W matrices
    # U shape: (n_target * m, rank)
    # V shape: (m * p, rank)
    # W shape: (p * n_target, rank)
    
    U = [[0] * rank for _ in range(n_target * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n_target)]
    
    for c, (i, j, k) in enumerate(filtered_terms):
        U[i * m + j][c] = 1
        V[j * p + k][c] = 1
        W[k * n_target + i][c] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
