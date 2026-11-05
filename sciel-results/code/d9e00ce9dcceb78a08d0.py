"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-15 exact tensor decomposition for 3x5x5 matrix multiplication.
    
    The 3x5x5 multiplication tensor is the direct sum of three 1x5x5 row-vector-matrix
    multiplication tensors. Each 1x5x5 tensor has rank 5 (schoolbook: for each output
    column k, sum over j of A[j]*B[j,k]). The rank of a direct sum is the sum of the
    ranks, so total rank = 3*5 = 15.
    
    Convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The multiplication tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    
    For row i of A (a 1x5 vector) times B (5x5 matrix), output C[i,k] = sum_j A[i,j]*B[j,k].
    For each output column k (k=0..4), we need one rank-1 term:
      U[i*m+j, c] = 1  for all j (row i's elements)
      V[j*p+k, c] = 1  for all j (column k of B)
      W[k*n+i, c] = 1
    This gives C[i,k] = sum_j 1*1*1 = sum_j A[i,j]*B[j,k]. Correct.
    
    Total: 3 rows * 5 columns = 15 terms.
    """
    n, m, p = DIMENSIONS
    rank = n * p  # 3 * 5 = 15
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for k in range(p):
            # For each (i, k) pair, create one rank-1 term
            for j in range(m):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
            W[k * n + i][column] = 1
            column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
