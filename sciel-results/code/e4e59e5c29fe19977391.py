import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    t0 = time.time()
    deadline = t0 + 130.0

    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def check_decomposition(U, V, W):
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(reconstructed, T, atol=1e-9)

    def to_lists(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Rank-8 decomposition:
    # For each j in {0..m-1}, the contribution is:
    #   sum_k e_{im+j} (over i) tensor e_{jp+k} tensor e_{kn+i}
    # = e_j tensor (sum_k e_{jp+k} tensor e_{2k})  [for i=0 part]
    # + e_{m+j} tensor (sum_k e_{jp+k} tensor e_{2k+1})  [for i=1 part]
    # Each of these is a rank-1 tensor, so total rank = 2*m = 8
    rank = 2 * m
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    for j in range(m):
        # Term for i=0: U=e_j, V=ones at positions jp..jp+p-1, W=ones at even positions 0,2,...,2(p-1)
        col = 2 * j
        U[j, col] = 1.0
        V[j*p : j*p + p, col] = 1.0
        W[0 : p*n : 2, col] = 1.0  # positions 0, 2, 4, ..., 2*(p-1)

        # Term for i=1: U=e_{m+j}, V=ones at positions jp..jp+p-1, W=ones at odd positions 1,3,...,2p-1
        col = 2 * j + 1
        U[m + j, col] = 1.0
        V[j*p : j*p + p, col] = 1.0
        W[1 : p*n : 2, col] = 1.0  # positions 1, 3, 5, ..., 2p-1

    if not check_decomposition(U, V, W):
        # Fallback to trivial rank-48 decomposition
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        return to_lists(U, V, W)

    return to_lists(U, V, W)
