"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    # Schoolbook rank is n*m*p = 36. We try to find a lower-rank exact decomposition.
    # For this specific shape, the minimal rank is known to be less than 36.
    # We use a structured construction based on the known decomposition patterns.
    #
    # The multiplication tensor T has entries T[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
    # The bilinear form is: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor notation: T[i, j, k] = 1 for all valid indices (it's the identity-like tensor)
    #
    # Actually, the matrix multiplication tensor for (n x m) * (m x p) -> (n x p)
    # has the property that the tensor is T[i, j, k] where the multiplication is
    # C[i, k] = sum_j A[i, j] B[j, k]
    #
    # The standard schoolbook decomposition uses rank = n*m*p with each term
    # selecting one (i, j, k) triple.
    #
    # For 2x3 * 3x6, let's try rank 18 or lower using known algebraic identities.
    #
    # One approach: use the fact that we can share computations across different i or k.
    # For example, if we have two rows i1, i2 and they share the same j-dependence,
    # we can factor out.
    #
    # Let's try a direct construction. The key insight is that for matrix multiplication,
    # the tensor is actually a "slice" structure. We can potentially reduce rank by
    # combining terms that differ only in the output index.
    #
    # For 2x3 * 3x6:
    # C[0, k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1, k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    #
    # for k = 0, 1, 2, 3, 4, 5
    #
    # Total of 12 output entries, each is a sum of 3 products.
    #
    # Let's try rank 18: we can potentially pair up the k indices.
    # For each j and each i, we have 6 k values. If we could combine two k's per term...
    #
    # Actually, let's just use the schoolbook but check if there's a known lower bound.
    # The Strassen algorithm works for 2x2*2x2 with rank 7 < 8.
    # For our case, let's try to construct a rank 18 decomposition.
    #
    # Strategy: For each j (0,1,2) and each pair of k values, create a term that
    # handles both k values simultaneously using linear combinations.
    #
    # But this requires the A and B matrices to have specific structure, which they don't.
    #
    # Let me just return the schoolbook decomposition. Rank = 36.
    n, m, p = DIMENSIONS
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
