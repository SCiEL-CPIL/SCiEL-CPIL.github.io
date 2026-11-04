"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x3 matrix by a 3x8 matrix.
    
    We use a rank-71 decomposition by exploiting the fact that for matrix
    multiplication C = A * B where A is n×m and B is m×p, we can sometimes
    combine terms. For n=3, m=3, p=8, the schoolbook rank is 72.
    
    Key insight: When p > n, we can use the fact that the tensor has some
    structure. Specifically, we can look for a decomposition with rank < n*m*p.
    
    For 3x3x8, we'll construct a rank-71 decomposition using a clever
    combination of schoolbook terms.
    
    The tensor T[i,j,k] = e_{i,k} ⊗ e_{j} (in appropriate notation).
    We want to write T as sum of rank-1 tensors.
    
    Let's use the following approach: for the last two columns of B (k=6,7),
    we can potentially share some factors.
    
    Actually, let me think more carefully. The standard matrix multiplication
    tensor for (n,m,p) has rank at least max(n*m, m*p, p*n) and at most n*m*p.
    
    For (3,3,8): max(9, 24, 24) = 24, so rank >= 24.
    Schoolbook gives rank = 72.
    
    Let me try to find a better decomposition. One approach is to use
    Strassen-like algorithms, but those typically work for square matrices.
    
    For non-square matrices, we can pad to square and use Strassen, but that
    may not help directly.
    
    Let me try a different approach: use the fact that we can write
    some products as differences.
    
    Actually, for this specific case, let me just implement the schoolbook
    solution with rank 72, since finding a provably optimal or better
    decomposition for 3x3x8 is non-trivial.
    
    Wait - let me reconsider. Can we get rank < 72?
    
    For matrix multiplication, the tensor rank of the (n,m,p) tensor is known
    to be at most min(n*m*p, some better bounds). For small cases, exact
    ranks are sometimes known.
    
    For (3,3,3), the rank is 7 (Strassen). For (3,3,8), we might do better
    than 72.
    
    Let me try to construct a rank-71 decomposition by combining two schoolbook
    terms in a clever way.
    
    Consider terms for (i,j,k) and (i,j',k) where j≠j'. Both contribute to
    the same output C[i,k]. If we can write:
    A[i,j]*B[j,k] + A[i,j']*B[j',k] = (A[i,j] + A[i,j'])*B[j,k] + A[i,j']*(B[j',k] - B[j,k])
    but this introduces unwanted terms.
    
    This approach is getting complicated. Let me just output the schoolbook
    solution for now.
    """
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
