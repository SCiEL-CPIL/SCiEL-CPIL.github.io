"""Rank-minimal tensor decomposition for 3x3 * 3x4 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    n, m, p = DIMENSIONS

    # Construct target tensor: T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition (rank = n*m*p = 36)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        T_r = np.zeros((n * m, m * p, p * n))
        for c in range(r):
            T_r += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return bool(np.max(np.abs(T_r - T)) < tol)

    # Attempt to find lower-rank decomposition via numerical optimization + rounding
    for r in range(20, rank_sb):
        if time.time() - start_time > DEADLINE - 5:
            break
        found = False
        for restart in range(8):
            if time.time() - start_time > DEADLINE - 5:
                break
            U = np.random.randn(n * m, r) * 0.3
            V = np.random.randn(m * p, r) * 0.3
            W = np.random.randn(p * n, r) * 0.3
            lr = 0.02
            for it in range(400):
                if time.time() - start_time > DEADLINE - 5:
                    break
                T_r = np.zeros((n * m, m * p, p * n))
                for c in range(r):
                    T_r += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
                res = T_r - T
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                for c in range(r):
                    gU[:, c] = 2.0 * np.einsum('abc,db,ec->ae', res, V[:, c], W[:, c])
                    gV[:, c] = 2.0 * np.einsum('abc,ac,dc->be', res, U[:, c], W[:, c])
                    gW[:, c] = 2.0 * np.einsum('abc,ab,ec->dc', res, U[:, c], V[:, c])
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                if np.max(np.abs(res)) < 1e-12:
                    break
            # Try rounding to half-integers
            Ur = np.round(U * 2.0) / 2.0
            Vr = np.round(V * 2.0) / 2.0
            Wr = np.round(W * 2.0) / 2.0
            if verify(Ur, Vr, Wr):
                best_U, best_V, best_W = Ur, Vr, Wr
                found = True
                break
        if found:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
