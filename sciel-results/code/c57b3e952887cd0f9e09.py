"""Matrix multiplication 2x4 x 4x8 - rank minimization."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    A = n*m; B = m*p; C = p*n
    for a in range(A):
        for b in range(B):
            for c in range(C):
                s = np.dot(U[a], V[b] * W[c])
                # expected
                expected = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if a == i*m+j and b == j*p+k and c == k*n+i:
                                expected = 1.0
                if abs(s - expected) > tol:
                    return False
    return True

def construct():
    start = time.time()
    deadline = start + 130.0
    n, m, p = DIMENSIONS
    A, B, C = n*m, m*p, p*n
    rank0 = n*m*p  # 64

    # Schoolbook
    U0 = np.zeros((A, rank0))
    V0 = np.zeros((B, rank0))
    W0 = np.zeros((C, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1.0
                V0[j*p+k, col] = 1.0
                W0[k*n+i, col] = 1.0
                col += 1

    # Build tensor support
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i*m+j, j*p+k, k*n+i)] = 1.0

    def _err(U, V, W):
        e = 0.0
        for (a,b,c), v in T.items():
            s = np.dot(U[a], V[b]*W[c])
            e += (s-v)**2
        for a in range(A):
            for b in range(B):
                for c in range(C):
                    if (a,b,c) not in T:
                        s = np.dot(U[a], V[b]*W[c])
                        e += s*s
        return e

    # Try to reduce rank using ALS with random restarts
    best_rank = rank0
    best = (U0, V0, W0)
    lb = max(np.linalg.matrix_rank(np.zeros((A*B,C)), tol=1e-10), 1)
    # compute matricization rank
    TAC = np.zeros((A*B, C))
    for (a,b,c),v in T.items():
        TAC[a*B+b, c] = v
    lb = max(lb, np.linalg.matrix_rank(TAC, tol=1e-10))

    for r in range(lb, rank0):
        if time.time() > deadline - 15:
            break
        found = False
        for seed in range(3):
            if time.time() > deadline - 15:
                break
            rng = np.random.RandomState(seed)
            U = U0[:, :r].copy() + rng.randn(A, r)*0.05
            V = V0[:, :r].copy() + rng.randn(B, r)*0.05
            W = W0[:, :r].copy() + rng.randn(C, r)*0.05
            for it in range(300):
                if time.time() > deadline - 15:
                    break
                # ALS: update U
                for a in range(A):
                    for rr in range(r):
                        pass
                # simple: gradient-free check + perturb
                e = _err(U, V, W)
                U += rng.randn(A, r)*0.002
                V += rng.randn(B, r)*0.002
                W += rng.randn(C, r)*0.002
                e2 = _err(U, V, W)
                if e2 > e:
                    U -= rng.randn(A, r)*0.002
                    V -= rng.randn(B, r)*0.002
                    W -= rng.randn(C, r)*0.002
                if e < 1e-10:
                    found = True
                    break
            if found:
                break
        if found:
            Ur = [[round(x*2)/2 for x in row] for row in U]
            Vr = [[round(x*2)/2 for x in row] for row in V]
            Wr = [[round(x*2)/2 for x in row] for row in W]
            if _verify(Ur, Vr, Wr, n, m, p, 1e-6):
                return {"U": Ur, "V": Vr, "W": Wr}
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, 1e-9):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
