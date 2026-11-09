"""Matrix multiplication 3x4x6 rank-minimal decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    # Try to find a rank < 72 decomposition
    # The tensor T[(i,j),(j,k),(k,i)] = delta (it's 1 when indices match)
    # Schoolbook rank = n*m*p = 72
    #
    # Key insight: For 3x4x6, we can potentially use the structure where
    # the 4 middle dimension allows some sharing.
    #
    # Approach: Use the fact that for each pair (i,k), the contribution
    # is sum_j A[i,j]*B[j,k]. We can try to represent this with fewer terms
    # by using a "mixed" decomposition.
    #
    # Let's try rank = 64 first, then search lower.
    # Actually, let me think: the rank of [n,m,p] matmul tensor is at least
    # max(n*m, m*p, p*n) = max(12, 24, 18) = 24.
    # And at most n*m*p = 72.
    #
    # For 3x4x6, let me try a structured decomposition.
    #
    # One approach: decompose the 4-dim middle into 2+2
    # A[i, 0:2] and A[i, 2:4], B[0:2, k] and B[2:4, k]
    # C[i,k] = A[i,0:2]@B[0:2,k] + A[i,2:4]@B[2:4,k]
    # Each 3x2x6 matmul has rank 3*2*6 = 36
    # Total: 72. No improvement.
    #
    # Try 3x1x6 + 3x3x6:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1:4]@B[1:4,k]
    # 3x1x6: rank 3*1*6 = 18
    # 3x3x6: rank 3*3*6 = 54
    # Total: 72. Still no improvement.
    #
    # The issue is that simple block decompositions don't help.
    # We need a more clever approach.
    #
    # Let me try to use the fact that 3x4 can be embedded in a way that
    # shares structure. Actually, for small dimensions, the optimal rank
    # might indeed be 72, or slightly less.
    #
    # Let me try a direct search: assume rank r, and see if we can find
    # U, V, W with the constraint that U@V^T (element-wise per k,i) = W
    #
    # The constraint is: for all c, the matrix U[:,c] (12x1) outer V[:,c] (24x1)
    # when reshaped and contracted gives the right W[:,c]
    #
    # More precisely: W[(k,i),c] = sum_j U[(i,j),c] * V[(j,k),c]
    # This is a system of equations in the entries of U, V, W.
    #
    # For a given rank r, this is a system where we need to find rank-r
    # "matrices" (in a generalized sense) that decompose the tensor.
    #
    # Let me just return the schoolbook for now, as finding a lower rank
    # for 3x4x6 is non-trivial and may require significant search time.
    #
    # Actually, let me try one more thing: use the identity
    # sum_j A[i,j]*B[j,k] = sum over some shared structure
    #
    # For n=3, m=4, p=6: the rank is known in literature?
    # [3,4,6] matrix multiplication rank...
    #
    # I'll return schoolbook rank 72 as a safe valid answer.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
