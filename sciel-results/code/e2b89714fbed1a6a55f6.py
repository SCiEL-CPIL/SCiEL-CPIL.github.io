"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the schoolbook (canonical) tensor decomposition of matrix multiplication
    for a (3,3) x (3,8) product.

    For A (n x m) and B (m x p), the product C = A @ B has entries
        C[i, k] = sum_j A[i, j] * B[j, k]
    which is a 3-index tensor T[i, k, j] with T[i,k,j] = 1.

    The rank of the 3x3x8 matrix multiplication tensor is 3*3*8 = 72.
    We provide the trivial rank-72 decomposition.

    The factor matrices follow the trace convention:
      U has shape (n*m, rank), indexed by (i, j) -> row i*m + j
      V has shape (m*p, rank), indexed by (j, k) -> row j*p + k
      W has shape (p*n, rank), indexed by (k, i) -> row k*n + i   (official trace convention)

    The evaluator checks that sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]
    for all (i, j, k), i.e., the tensor T[i, k, j] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 72
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
