import time
import random
import math
import numpy as np

DIMENSIONS = [4, 5, 5]
N, M, P = DIMENSIONS
DEADLINE = 150.0
SEED = 123456789


def _target_tensor():
    T = np.zeros((N, M, P), dtype=np.int64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i, j, k] = 1
    return T


TARGET = _target_tensor()


def _cp_decompose(U, V, W):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    R = U.shape[1]
    T = np.zeros((N, M, P), dtype=np.float64)
    for r in range(R):
        T += np.multiply.outer(np.multiply.outer(U[:, r], V[:, r]), W[:, r])
    return T


def _validate(U, V, W):
    T = _cp_decompose(U, V, W)
    if not np.allclose(T, TARGET, rtol=1e-9, atol=1e-9):
        return False
    return True


def _round_half(U, V, W):
    U = np.round(2.0 * np.asarray(U, dtype=np.float64)) / 2.0
    V = np.round(2.0 * np.asarray(V, dtype=np.float64)) / 2.0
    W = np.round(2.0 * np.asarray(W, dtype=np.float64)) / 2.0
    return U, V, W


def _is_half_integer(U, V, W):
    for Mx in (U, V, W):
        if not np.allclose(Mx, np.round(2.0 * Mx) / 2.0, atol=1e-6):
            return False
    return True


def _matmul_vec(A, x):
    return A.dot(x)


def _matmul_vecT(A, x):
    return A.T.dot(x)


def _local_refine(U, V, W, T, iters=40, step=0.05):
    U = np.array(U, dtype=np.float64, copy=True)
    V = np.array(V, dtype=np.float64, copy=True)
    W = np.array(W, dtype=np.float64, copy=True)
    R = U.shape[1]
    nU, nV, nW = U.shape[0], V.shape[0], W.shape[0]
    best = (U.copy(), V.copy(), W.copy())
    best_err = float('inf')
    rng = random.Random(SEED + 1)

    for it in range(iters):
        # gradient descent on squared error
        err = 0.0
        for r in range(R):
            e = T - np.multiply.outer(np.multiply.outer(U[:, r], V[:, r]), W[:, r])
            err += np.sum(e * e)
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if err < 1e-12:
            break

        # compute gradients
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        for r in range(R):
            e = T - np.multiply.outer(np.multiply.outer(U[:, r], V[:, r]), W[:, r])
            gU[:, r] += 2.0 * np.tensordot(e, np.outer(V[:, r], W[:, r]), axes=([0, 1], [0, 1]))
            gV[:, r] += 2.0 * np.tensordot(e, np.outer(U[:, r], W[:, r]), axes=([0, 2], [0, 1]))
            gW[:, r] += 2.0 * np.tensordot(e, np.outer(U[:, r], V[:, r]), axes=([0, 1], [0, 1]))

        # adaptive step
        lr = step
        for _ in range(6):
            U2 = U - lr * gU
            V2 = V - lr * gV
            W2 = W - lr * gW
            err2 = 0.0
            for r in range(R):
                e = T - np.multiply.outer(np.multiply.outer(U2[:, r], V2[:, r]), W2[:, r])
                err2 += np.sum(e * e)
            if err2 < err:
                U, V, W = U2, V2, W2
                err = err2
                break
            lr *= 0.5
    return best


def _try_rank(R, time_budget):
    """Try to find a rank-R decomposition. Returns (U,V,W) or None."""
    start = time.time()
    rng = random.Random(SEED + R)
    best = None
    best_err = float('inf')

    for attempt in range(200):
        if time.time() - start > time_budget:
            break
        # random init
        U = rng.standard_normal((N * M, R))
        V = rng.standard_normal((M * P, R))
        W = rng.standard_normal((P * N, R))
        # normalize to avoid overflow
        for r in range(R):
            nU = np.linalg.norm(U[:, r])
            nV = np.linalg.norm(V[:, r])
            nW = np.linalg.norm(W[:, r])
            if nU < 1e-9 or nV < 1e-9 or nW < 1e-9:
                continue
            U[:, r] /= nU
            V[:, r] /= nV
            W[:, r] /= nW

        U, V, W = _local_refine(U, V, W, TARGET, iters=20, step=0.05)
        err = 0.0
        for r in range(R):
            e = TARGET - np.multiply.outer(np.multiply.outer(U[:, r], V[:, r]), W[:, r])
            err += np.sum(e * e)
        if err < best_err:
            best_err = err
            best = (U, V, W)
        if err < 1e-10:
            break
        if time.time() - start > time_budget:
            break

    if best is None:
        return None
    U, V, W = best
    # round to half-integers
    U, V, W = _round_half(U, V, W)
    if _validate(U, V, W):
        return (U, V, W)
    return None


def construct():
    start = time.time()
    # Try ranks from low to high
    # Known lower bound: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # Schoolbook rank = 100
    min_rank = max(N * M, M * P, P * N)
    max_rank = N * M * P

    # First, get schoolbook as fallback
    rank_sb = N * M * P
    U_sb = np.zeros((N * M, rank_sb), dtype=np.int64)
    V_sb = np.zeros((M * P, rank_sb), dtype=np.int64)
    W_sb = np.zeros((P * N, rank_sb), dtype=np.int64)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U_sb[i * M + j, col] = 1
                V_sb[j * P + k, col] = 1
                W_sb[k * N + i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb

    # Try to find lower rank
    for R in range(min_rank, best_rank):
        if time.time() - start > DEADLINE - 10:
            break
        time_budget = min(20.0, DEADLINE - 10 - (time.time() - start))
        if time_budget < 5.0:
            break
        result = _try_rank(R, time_budget)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = R
            break  # found minimal, stop

    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    _validate(np.array(res["U"]), np.array(res["V"]), np.array(res["W"]))
    print("Valid")
