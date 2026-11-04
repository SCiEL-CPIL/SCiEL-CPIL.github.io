"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 125

    n, m, p = DIMENSIONS

    # Build target tensor T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, n * p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition
    rank_school = n * m * p
    U_s = np.zeros((n * m, rank_school))
    V_s = np.zeros((m * p, rank_school))
    W_s = np.zeros((n * p, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1
                V_s[j * p + k, col] = 1
                W_s[k * n + i, col] = 1
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_s, V_s, W_s

    def check(U, V, W, tol=1e-10):
        max_err = 0.0
        for a in range(n * m):
            for b in range(m * p):
                recon = W.T @ (U[a] * V[b])
                e = np.max(np.abs(recon - T[a, b]))
                if e > max_err:
                    max_err = e
        return max_err < tol

    def to_frac(M):
        return [[Fraction(str(round(float(v) * 2) / 2)) for v in row] for row in M]

    # Compute matrix rank lower bounds
    lb = max(
        np.linalg.matrix_rank(T.reshape(n * p, -1), tol=1e-10),
        np.linalg.matrix_rank(T.reshape(n * m, -1), tol=1e-10),
        np.linalg.matrix_rank(T.reshape(m * p, -1), tol=1e-10)
    )

    # Search for lower rank using ALS with random restarts
    np.random.seed(0)
    min_r = max(int(lb), 50)

    for r in range(min_r, rank_school + 1):
        if time.time() > deadline - 15:
            break
        found = False
        for trial in range(8):
            if time.time() > deadline - 15:
                break
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(n * p, r)

            for it in range(300):
                if time.time() > deadline - 15:
                    break
                # Solve for W given U, V
                A = np.array([np.outer(U[:, l], V[:, l]).flatten() for l in range(r)]).T
                T1 = T.reshape(n * m * m * p, n * p)
                try:
                    W = np.linalg.lstsq(A, T1, rcond=None)[0].T
                except:
                    pass
                # Solve for U given V, W
                B = np.array([np.outer(V[:, l], W[:, l]).flatten() for l in range(r)]).T
                T2 = T.reshape(n * m, m * p * n * p)
                try:
                    U = np.linalg.lstsq(B, T2, rcond=None)[0].T
                except:
                    pass
                # Solve for V given U, W
                C = np.array([np.outer(U[:, l], W[:, l]).flatten() for l in range(r)]).T
                T3 = T.reshape(m * p, n * m * n * p)
                try:
                    V = np.linalg.lstsq(C, T3, rcond=None)[0].T
                except:
                    pass

            if check(U, V, W, 1e-8) and check(U, V, W, 1e-10):
                Ur = np.round(U * 2) / 2
                Vr = np.round(V * 2) / 2
                Wr = np.round(W * 2) / 2
                if check(Ur, Vr, Wr, 1e-9):
                    if r < best_rank:
                        best_rank, best_U, best_V, best_W = r, Ur, Vr, Wr
                    found = True
                    break
        if found:
            continue

    return {"U": to_frac(best_U), "V": to_frac(best_V), "W": to_frac(best_W)}
# EVOLVE-BLOCK-END
