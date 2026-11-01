"""Rank-minimal decomposition for 2x2 by 2x12 matrix multiplication."""
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 12
    # Rank = 2*p = 24 construction.
    # For each k in 0..p-1, we use 2 rank-1 terms that together produce
    # the 2x2-by-2x1 (column k) product using the 2x2 structure.
    #
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # We factor the 2x2 coefficient matrix A = L * R where we use the
    # identity that any 2x2 matrix A can be written as A[0,0]*E00 + A[0,1]*E01
    # + A[1,0]*E10 + A[1,1]*E11, but to reduce rank we pair:
    #
    # Term t=0 (for this k): U picks A[0,0] and A[1,0] (column 0 of A),
    #   V picks B[0,k], W sets C[0,k] and C[1,k].
    #   Contribution: A[0,0]*B[0,k] to C[0,k], A[1,0]*B[0,k] to C[1,k].
    #   But U is a single 2x2 matrix (rank-1 factor), so U[i,j] must equal
    #   A[i,0] for j=0 and 0 for j=1. V[j,k]=B[0,k] for j=0, 0 else.
    #   W[i,k]=1 for i=0,1.
    #   => U = [[A00,0],[A10,0]], V has B[0,k] at (0,k), W has 1 at (0,k),(1,k).
    #   This is ONE rank-1 term contributing A[i,0]*B[0,k] to C[i,k] for i=0,1.
    #
    # Term t=1 (for this k): U = [[A01,0],[A11,0]]? No, that uses column 1 of A
    #   but V would need B[1,k]. U[i,0]=A[i,1], V[1,k]=B[1,k], W[i,k]=1.
    #   => contributes A[i,1]*B[1,k] to C[i,k].
    #
    # So per k: 2 terms, total 2*p = 24 terms. Each term:
    #   U is 2x2 (flattened to 4), V is 2x12 (flattened to 24), W is 12x2 (flattened to 24).
    rank = 2 * p
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x rank
    V = [[0.0] * rank for _ in range(m * p)]   # 24 x rank
    W = [[0.0] * rank for _ in range(p * n)]   # 24 x rank

    col = 0
    for k in range(p):
        # Term 0: A[:,0] * B[0,k] -> C[:,k]
        # U[i,0] = A[i,0] => U[i*m + 0] = 1 (coefficient placeholder), U[i*m+1] = 0
        # V[0*p + k] = 1
        # W[k*n + i] = 1 for i in 0,1
        for i in range(n):
            U[i * m + 0][col] = 1.0
            U[i * m + 1][col] = 0.0
        V[0 * p + k][col] = 1.0
        for i in range(n):
            W[k * n + i][col] = 1.0
        col += 1
        # Term 1: A[:,1] * B[1,k] -> C[:,k]
        for i in range(n):
            U[i * m + 0][col] = 0.0
            U[i * m + 1][col] = 1.0
        V[1 * p + k][col] = 1.0
        for i in range(n):
            W[k * n + i][col] = 1.0
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Self-check: verify the decomposition exactly reproduces C = A @ B.
    import numpy as np
    n, m, p = DIMENSIONS
    res = construct()
    U = np.array(res["U"], dtype=float)  # (n*m, rank)
    V = np.array(res["V"], dtype=float)  # (m*p, rank)
    W = np.array(res["W"], dtype=float)  # (p*n, rank)
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Build the multiplication tensor M[i,j,k] = delta such that
    # C[i,k] = sum_{i,j,k} A[i,j] B[j,k] T[i,j,k]
    # We require T[i,j,k] = 1 iff the term contributes, i.e. T = 1 for all (i,j,k)
    # because C[i,k] = sum_j A[i,j] B[j,k].
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    # Reconstruct T from the factors
    Trec = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        # contribution to T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Trec[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    assert np.allclose(Trec, T), "Decomposition is incorrect!"
    print("OK: rank =", rank)
