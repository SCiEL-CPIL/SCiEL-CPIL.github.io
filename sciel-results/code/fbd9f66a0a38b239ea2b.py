"""Tensor rank minimization for 4x5x6 matrix multiplication."""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct factor matrices for matrix multiplication tensor.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    The decomposition satisfies: sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c] * 0 + ...
    Actually, the correct condition is:
    For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] = (1 if W[k*n+i, c] is the coefficient)
    
    More precisely: the matrix multiplication tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c]
    must equal the standard basis, i.e., T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    
    The standard schoolbook: for each (i,j,k), one rank-1 term with U[i*m+j]=1, V[j*p+k]=1, and
    the output index is k*n+i. So W[k*n+i, c] = 1.
    
    We return the schoolbook decomposition of rank n*m*p = 120.
    """
    n, m, p = DIMENSIONS
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
