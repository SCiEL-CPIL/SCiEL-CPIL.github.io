"""Matrix multiplication 2x2 by 2x5: search for a rank-minimal exact decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 5]


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _tensors(U, V, W):
    T1 = np.zeros((U.shape[0], V.shape[0]))
    T2 = np.zeros((V.shape[0], W.shape[0]))
    for r in range(U.shape[1]):
        T1 += np.outer(U[:, r], V[:, r])
        T2 += np.outer(V[:, r], W[:, r])
    return T1, T2


def _residual(U, V, W):
    T1, T2 = _tensors(U, V, W)
    return float(np.sum(T1 * T1) + np.sum(T2 * T2))


def _als(U, V, W, iters=120, tol=1e-11, max_norm=1e6):
    R = U.shape[1]
    T1 = np.zeros((U.shape[0], V.shape[0]))
    T2 = np.zeros((V.shape[0], W.shape[0]))
    for r in range(R):
        T1 += np.outer(U[:, r], V[:, r])
        T2 += np.outer(V[:, r], W[:, r])
    best = _residual(U, V, W)
    for it in range(iters):
        # Update U_r: min ||T1 - U_r V_r^T|| -> U_r = T1 @ V_r / (V_r^T V_r)
        for r in range(R):
            v = V[:, r]
            U[:, r] = T1 @ v / (v @ v + 1e-12)
        T1 = np.zeros((U.shape[0], V.shape[0]))
        for r in range(R):
            T1 += np.outer(U[:, r], V[:, r])
        # Update W_r: min ||T2 - V_r W_r^T|| -> W_r = T2^T @ V_r / (V_r^T V_r)
        for r in range(R):
            v = V[:, r]
            W[:, r] = T2.T @ v / (v @ v + 1e-12)
        T2 = np.zeros((V.shape[0], W.shape[0]))
        for r in range(R):
            T2 += np.outer(V[:, r], W[:, r])
        # Update V_r: V_r = (T1^T U_r + T2 W_r) / (||U_r||^2 + ||W_r||^2)
        for r in range(R):
            denom = (U[:, r] @ U[:, r] + W[:, r] @ W[:, r] + 1e-12)
            V[:, r] = (T1.T @ U[:, r] + T2 @ W[:, r]) / denom
        res = _residual(U, V, W)
        if res < best:
            best = res
        if best < tol:
            break
        if max(np.linalg.norm(U), np.linalg.norm(V), np.linalg.norm(W)) > max_norm:
            break
    return U, V, W, best


def _verify(Uq, Vq, Wq):
    n, m, p = DIMENSIONS
    T1, T2 = _tensors(Uq, Vq, Wq)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T1[i * m + j, j * p + k] - 1) > 1e-9:
                    return False
                if abs(T2[j * p + k, k * n + i] - 1) > 1e-9:
                    return False
    return True


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    U0, V0, W0 = _schoolbook()
    rng = np.random.default_rng(0)

    def try_rank(R, restarts):
        for restart in range(restarts):
            if time.time() > deadline - 5:
                return None
            if R <= U0.shape[1]:
                idx = rng.choice(U0.shape[1], size=R, replace=False)
                U = U0[:, idx].copy()
                V = V0[:, idx].copy()
                W = W0[:, idx].copy()
            else:
                U = rng.normal(size=(n * m, R))
                V = rng.normal(size=(m * p, R))
                W = rng.normal(size=(p * n, R))
            U = U + 0.01 * rng.normal(size=(n * m, R))
            V = V + 0.01 * rng.normal(size=(m * p, R))
            W = W + 0.01 * rng.normal(size=(p * n, R))
            U, V, W, res = _als(U, V, W, iters=100, tol=1e-11)
            if res < 1e-8:
                Uq = np.round(2 * U) / 2
                Vq = np.round(2 * V) / 2
                Wq = np.round(2 * W) / 2
                if _verify(Uq, Vq, Wq):
                    return {"U": Uq.tolist(), "V": Vq.tolist(), "W": Wq.tolist()}
        return None

    for R in [17, 18, 19, 20]:
        restarts = {17: 40, 18: 20, 19: 10, 20: 5}[R]
        res = try_rank(R, restarts)
        if res is not None:
            return res

    # fallback to schoolbook
    Uq = np.round(2 * U0) / 2
    Vq = np.round(2 * V0) / 2
    Wq = np.round(2 * W0) / 2
    return {"U": Uq.tolist(), "V": Vq.tolist(), "W": Wq.tolist()}
