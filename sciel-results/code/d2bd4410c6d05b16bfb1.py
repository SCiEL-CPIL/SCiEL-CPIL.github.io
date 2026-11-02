"""Rewritten program for the matrix multiplication tensor decomposition task."""
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (known) decomposition for 3x4 * 4x6 matrix multiplication.
    We use the schoolbook decomposition with rank 72 (n*m*p).
    The tensor identity is: sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r] (for the appropriate r index)
    Actually, the standard form is:
    (A*B)[i,k] = sum_{j} A[i,j] B[j,k]
    Decomposition: (A*B)[i,k] = sum_{r} W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] ??? No.
    The standard bilinear decomposition is:
    (A*B)[i,k] = sum_{r} M_r(i,k) * L_r(A) * R_r(B)
    where L_r(A) = sum_{i,j} U[i*m+j, r] A[i,j]
    and R_r(B) = sum_{j,k} V[j*p+k, r] B[j,k]
    and the output coefficient is W[k*n+i, r].
    So: sum_{r} W[k*n+i, r] * (sum_{j} U[i*m+j, r] A[i,j] ... wait, L_r depends on i,j.
    Let's stick to the schoolbook:
    For each (i,j,k), we have a term A[i,j] * B[j,k].
    r corresponds to (i,j,k).
    U[i*m+j, r] = 1 if r encodes (i,j,k) and the first part is (i,j).
    V[j*p+k, r] = 1 if r encodes (i,j,k) and the second part is (j,k).
    W[k*n+i, r] = 1 if r encodes (i,j,k) and the output index is (i,k).
    This gives rank n*m*p = 72.
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
