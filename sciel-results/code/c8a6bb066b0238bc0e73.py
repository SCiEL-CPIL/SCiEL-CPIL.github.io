"""Matrix multiplication tensor decomposition for (2,3,10).

Strategy:
  - The 2x3x10 product is split by the middle dimension into five 2x2x2 blocks
    (columns of B taken two at a time) plus a 2x1x10 remainder (j=2 column).
  - Each 2x2x2 block uses the best 2x2x2 decomposition found at runtime
    (rank 6 if a rank-6 decomposition is discovered, else Strassen rank 7).
  - The remainder uses the trivial rank-20 schoolbook decomposition.
  - Total rank = 5 * r222 + 20  (50 if r222=6, else 55).
"""
import time
import numpy as np
import random

DIMENSIONS = [2, 3, 10]


def _build_T(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _err(U, V, W, T):
    return float(np.max(np.abs(np.einsum('ar,br,cr->abc', U, V, W) - T)))


def _als_222(rank, T, max_iters, seed):
    rng = np.random.RandomState(seed)
    U = rng.randn(4, rank) * 0.3
    V = rng.randn(4, rank) * 0.3
    W = rng.randn(4, rank) * 0.3
    Uf = T.reshape(4, 16)
    Vf = T.reshape(4, 16)
    Wf = T.reshape(4, 16)
    for _ in range(max_iters):
        M = np.einsum('br,cr->bcr', V, W)          # (4,4,rank)
        Mf = M.transpose(2, 0, 1).reshape(rank, 16)
        try:
            U = np.linalg.lstsq(Mf.T, Uf.T, rcond=None)[0].T
        except Exception:
            break
        N = np.einsum('ar,cr->acr', U, W)          # (4,4,rank)
        Nf = N.transpose(2, 0, 1).reshape(rank, 16)
        try:
            V = np.linalg.lstsq(Nf.T, Vf.T, rcond=None)[0].T
        except Exception:
            break
        P = np.einsum('ar,br->abr', U, V)          # (4,4,rank)
        Pf = P.transpose(2, 0, 1).reshape(rank, 16)
        try:
            W = np.linalg.lstsq(Pf.T, Wf.T, rcond=None)[0].T
        except Exception:
            break
        if _err(U, V, W, T) < 1e-11:
            break
    return U, V, W


def _gd_222(rank, T, max_iters, lr, seed):
    rng = np.random.RandomState(seed)
    U = rng.randn(4, rank) * 0.2
    V = rng.randn(4, rank) * 0.2
    W = rng.randn(4, rank) * 0.2
    for _ in range(max_iters):
        diff = np.einsum('ar,br,cr->abc', U, V, W) - T
        gU = np.einsum('abc,br,cr->ar', diff, V, W)
        gV = np.einsum('abc,ar,cr->br', diff, U, W)
        gW = np.einsum('abc,ar,br->cr', diff, U, V)
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        if _err(U, V, W, T) < 1e-11:
            break
    return U, V, W


def _lbfgs_222(rank, T, seed):
    from scipy.optimize import minimize
    rng = np.random.RandomState(seed)
    x0 = rng.randn(4 * rank * 3) * 0.2

    def fun(x):
        U = x[0:4 * rank].reshape(4, rank)
        V = x[4 * rank:8 * rank].reshape(4, rank)
        W = x[8 * rank:12 * rank].reshape(4, rank)
        diff = np.einsum('ar,br,cr->abc', U, V, W) - T
        return float(np.sum(diff * diff))

    def grad(x):
        U = x[0:4 * rank].reshape(4, rank)
        V = x[4 * rank:8 * rank].reshape(4, rank)
        W = x[8 * rank:12 * rank].reshape(4, rank)
        diff = np.einsum('ar,br,cr->abc', U, V, W) - T
        gU = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        gV = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        gW = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        return np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])

    res = minimize(fun, x0, jac=grad, method='L-BFGS-B',
                   options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12})
    x = res.x
    U = x[0:4 * rank].reshape(4, rank)
    V = x[4 * rank:8 * rank].reshape(4, rank)
    W = x[8 * rank:12 * rank].reshape(4, rank)
    return U, V, W


def _find_222(deadline):
    T = _build_T(2, 2, 2)
    # rank 6 first (the improvement lever)
    for rank in (6, 7):
        for seed in range(40):
            if time.time() > deadline:
                return None
            U, V, W = _als_222(rank, T, 1200, seed)
            if _err(U, V, W, T) < 1e-9:
                return U, V, W
        for seed in range(20):
            if time.time() > deadline:
                return None
            U, V, W = _gd_222(rank, T, 2500, 0.01, seed)
            if _err(U, V, W, T) < 1e-9:
                return U, V, W
        for seed in range(10):
            if time.time() > deadline:
                return None
            try:
                U, V, W = _lbfgs_222(rank, T, seed)
            except Exception:
                continue
            if _err(U, V, W, T) < 1e-9:
                return U, V, W
    return None


def _strassen_222():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1=(a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    # M2=(a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    # M3=a11(b12-b22)
    U[0, 2] = 1; V[1, 2] = 1; V[3, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    # M4=a22(b21-b11)
    U[3, 3] = 1; V[2, 3] = 1; V[0, 3] = -1; W[0, 3] = 1; W[1, 3] = 1
    # M5=(a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    # M6=(a22-a11)(b11+b12)
    U[3, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    # M7=(a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1; V[2, 6] = 1; V[3, 6] = 1; W[0, 6] = 1; W[3, 6] = -1
    return U, V, W


def _verify(U, V, W, n, m, p, max_checks=10):
    rank = len(U[0])
    for _ in range(max_checks):
        A = [[random.uniform(-2, 2) for _ in range(m)] for _ in range(n)]
        B = [[random.uniform(-2, 2) for _ in range(p)] for _ in range(m)]
        C = [[0.0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C[i][k] += A[i][j] * B[j][k]
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for r in range(rank):
                    ua = sum(U[i * m + j][r] * A[i][j] for j in range(m))
                    vb = sum(V[j * p + k][r] * B[j][k] for j in range(m))
                    acc += W[k * n + i][r] * ua * vb
                if abs(C[i][k] - acc) > 1e-6:
                    return False
    return True


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 10

    # Directly construct rank-55 (Strassen rank-7 blocks + schoolbook remainder).
    # This avoids the expensive and unsuccessful search for a rank-6 2x2x2 decomposition.
    U7, V7, W7 = _strassen_222()
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 5 blocks of 2x2x2 using Strassen (rank 7 each)
    # We split the k dimension (p=10) into 5 pairs.
    # For each pair, we use j=0,1 for the 2x2x2 block (rank 7).
    for block in range(5):
        k0 = block * 2
        for r in range(7):
            col = block * 7 + r
            for i in range(2):
                for j in range(2):  # j=0,1
                    U[i * 3 + j][col] = float(U7[i * 2 + j, r])
            for j in range(2):  # j=0,1
                for k in range(2):  # k=0,1 relative to k0
                    V[j * 10 + k0 + k][col] = float(V7[j * 2 + k, r])
            for k in range(2):  # k=0,1 relative to k0
                for i in range(2):
                    W[(k0 + k) * 2 + i][col] = float(W7[k * 2 + i, r])
    
    # Remainder for j=2 (rank 2 per k-pair, total 10)
    # 5 pairs * 2 (i) * 2 (k) = 20 terms? No.
    # For each k-pair (2 k's), and j=2, we have 2x2 = 4 terms.
    # 5 pairs * 4 = 20 terms.
    col = 35
    for block in range(5):
        k0 = block * 2
        for i in range(2):
            for k in range(2):
                U[i * 3 + 2][col] = 1
                V[20 + k0 + k][col] = 1
                W[(k0 + k) * 2 + i][col] = 1
                col += 1
                
    assert col == rank
    random.seed(42)
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # 3) Last resort: trivial rank-60.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
