import time
import random
import numpy as np

_DIMS = (2, 4, 4)
_T = None


def _build_T():
    n, m, p = _DIMS
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i
                T[x, y, z] = 1.0
    return T


def _verify(U, V, W, T, trials=3):
    """Correct evaluator check: reconstruct C = A@B from the tensor factors."""
    n, m, p = _DIMS
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rng = np.random.RandomState(0)
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        Cexp = A @ B
        Ccomp = np.zeros((n, p))
        for r in range(rank):
            ur = np.dot(A.ravel(), U[:, r])
            vr = np.dot(B.ravel(), V[:, r])
            Ccomp += (ur * vr) * W[:, r].reshape(p, n)
        if not np.allclose(Ccomp, Cexp, atol=1e-8):
            return False
    return True


def _round_half(X):
    return np.round(X * 2.0) / 2.0


def _find_redundant(U, V, W, T, deadline, tol=1e-6):
    """Find one removable column (a linear combination of the others) or None."""
    n, m, p = _DIMS
    R = U.shape[1]
    if R < 2:
        return None
    # rank-1 tensors flattened to vectors of length (n*m)*(m*p)*(p*n)
    T1 = np.einsum('ir,jr,kr->ijk', U, V, W).reshape(R, -1)
    for c in range(R):
        if time.time() > deadline:
            return None
        A = np.delete(T1, c, axis=0)          # (R-1, D)
        b = T1[c]
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = np.linalg.norm(b - A @ x)
        if res < tol:
            nU = np.delete(U, c, axis=1)
            nV = np.delete(V, c, axis=1)
            nW = np.delete(W, c, axis=1)
            nU = _round_half(nU)
            nV = _round_half(nV)
            nW = _round_half(nW)
            if _verify(nU, nV, nW, T, trials=3):
                return c
    return None


def _strassen_28():
    n, m, p = _DIMS
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def add_block(co, X, Y, Z):
        # M1
        U[X[0], co + 0] += 1; U[X[3], co + 0] += 1
        V[Y[0], co + 0] += 1; V[Y[3], co + 0] += 1
        W[Z[0], co + 0] += 1; W[Z[3], co + 0] += 1
        # M2
        U[X[2], co + 1] += 1; U[X[3], co + 1] += 1
        V[Y[0], co + 1] += 1
        W[Z[2], co + 1] += 1; W[Z[3], co + 1] -= 1
        # M3
        U[X[0], co + 2] += 1
        V[Y[1], co + 2] += 1; V[Y[3], co + 2] -= 1
        W[Z[1], co + 2] += 1; W[Z[3], co + 2] += 1
        # M4
        U[X[3], co + 3] += 1
        V[Y[2], co + 3] += 1; V[Y[0], co + 3] -= 1
        W[Z[0], co + 3] += 1; W[Z[2], co + 3] += 1
        # M5
        U[X[0], co + 4] += 1; U[X[1], co + 4] += 1
        V[Y[3], co + 4] += 1
        W[Z[0], co + 4] -= 1; W[Z[1], co + 4] += 1
        # M6
        U[X[2], co + 5] += 1; U[X[0], co + 5] -= 1
        V[Y[0], co + 5] += 1; V[Y[1], co + 5] += 1
        W[Z[3], co + 5] += 1
        # M7
        U[X[1], co + 6] += 1; U[X[3], co + 6] -= 1
        V[Y[2], co + 6] += 1; V[Y[3], co + 6] += 1
        W[Z[0], co + 6] += 1

    X1 = [0, 1, 4, 5]; Y1 = [0, 1, 4, 5]; Z1 = [0, 2, 1, 3]
    X2 = [2, 3, 6, 7]; Y2 = [8, 9, 12, 13]; Z2 = [0, 2, 1, 3]
    X3 = [0, 1, 4, 5]; Y3 = [2, 3, 6, 7]; Z3 = [4, 6, 5, 7]
    X4 = [2, 3, 6, 7]; Y4 = [10, 11, 14, 15]; Z4 = [4, 6, 5, 7]
    add_block(0, X1, Y1, Z1)
    add_block(7, X2, Y2, Z2)
    add_block(14, X3, Y3, Z3)
    add_block(21, X4, Y4, Z4)
    return U, V, W


def construct():
    global _T
    T = _build_T()
    _T = T
    U, V, W = _strassen_28()
    if not _verify(U, V, W, T, trials=3):
        # fallback: schoolbook rank 32 (always valid)
        n, m, p = _DIMS
        rank = n * m * p
        U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    start = time.time()
    deadline = start + 130.0   # safety margin under 150s

    # --- Phase 1: greedy removal of provably redundant columns ---
    U, V, W = best_U, best_V, best_W
    while time.time() < deadline:
        c = _find_redundant(U, V, W, T, deadline)
        if c is None:
            break
        U = np.delete(U, c, axis=1)
        V = np.delete(V, c, axis=1)
        W = np.delete(W, c, axis=1)
        best_U, best_V, best_W = U, V, W
        best_rank = U.shape[1]

    # --- Phase 2: deterministic half-integer mutations + removal ---
    rng = random.Random(12345)
    for _ in range(400):
        if time.time() > deadline:
            break
        U2, V2, W2 = best_U.copy(), best_V.copy(), best_W.copy()
        for _ in range(2):
            arr, rng_arr = (U2, V2, W2)[rng.randrange(3)]
            r = rng.randrange(arr.shape[0])
            c = rng.randrange(arr.shape[1])
            arr[r, c] += rng.choice([-0.5, 0.5])
        if not _verify(U2, V2, W2, T, trials=2):
            continue
        c = _find_redundant(U2, V2, W2, T, deadline)
        if c is not None:
            U2 = np.delete(U2, c, axis=1)
            V2 = np.delete(V2, c, axis=1)
            W2 = np.delete(W2, c, axis=1)
            if _verify(U2, V2, W2, T, trials=3):
                best_U, best_V, best_W = U2, V2, W2
                best_rank = U2.shape[1]
                # continue removing from the new base
                while time.time() < deadline:
                    c = _find_redundant(U2, V2, W2, T, deadline)
                    if c is None:
                        break
                    U2 = np.delete(U2, c, axis=1)
                    V2 = np.delete(V2, c, axis=1)
                    W2 = np.delete(W2, c, axis=1)
                    best_U, best_V, best_W = U2, V2, W2
                    best_rank = U2.shape[1]

    # final safety verify
    if not _verify(best_U, best_V, best_W, T, trials=3):
        U, V, W = _strassen_28()
        best_U, best_V, best_W = U, V, W

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
