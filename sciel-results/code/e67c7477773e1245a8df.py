"""Matrix multiplication 2x5x6: bounded randomized rank-minimal search with fallback."""
import time
import random
import math
import numpy as np

DIMENSIONS = [2, 5, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}

def _tens(U, V, W):
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = (U[i * m + j] * V[j * p + k] * W[k * n + i]).sum()
    return T

def _verify(U, V, W, tol=1e-8):
    n, m, p = DIMENSIONS
    T = _tens(U, V, W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i, j, k] - 1.0) > tol:
                    return False
    return True

def _rank_check(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    A = np.zeros((n * p, n * m))
    B = np.zeros((n * p, m * p))
    for r in range(rank):
        u = np.array(U[:, r]); v = np.array(V[:, r]); w = np.array(W[:, r])
        for i in range(n):
            for k in range(p):
                A[k * n + i, i * m + 0] = 0.0
                # A indexed [ (k,i), (i, j) ] with value u[i,j]*w[k,i]
        # build properly
    A = np.zeros((n * p, n * m))
    B = np.zeros((n * p, m * p))
    for r in range(rank):
        u = np.array(U[:, r]); v = np.array(V[:, r]); w = np.array(W[:, r])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A[k * n + i, i * m + j] += u[i * m + j] * w[k * n + i]
                    B[k * n + i, j * p + k] += u[i * m + j] * v[j * p + k]
    rA = np.linalg.matrix_rank(A, tol=tol)
    rB = np.linalg.matrix_rank(B, tol=tol)
    return min(rA, rB) >= rank

def _gen(rank, mode):
    n, m, p = DIMENSIONS
    if mode == 0:
        U = np.random.choice([-1.0, 1.0], size=(n * m, rank))
        V = np.random.choice([-1.0, 1.0], size=(m * p, rank))
        W = np.random.choice([-1.0, 1.0], size=(p * n, rank))
    elif mode == 1:
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        U /= (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)
        V /= (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
        W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    else:
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 150.0
    best = None
    n, m, p = DIMENSIONS
    max_rank = n * m * p
    for rank in range(2, max_rank + 1):
        if time.time() > deadline - 5.0:
            break
        trials = 3000
        for t in range(trials):
            if time.time() > deadline - 5.0:
                break
            mode = t % 3
            U, V, W = _gen(rank, mode)
            if _rank_check(U, V, W):
                if _verify(U, V, W):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    return best
    if best is not None:
        return best
    return _schoolbook()
