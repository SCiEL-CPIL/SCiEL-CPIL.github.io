"""Rewritten search for a rank-minimal exact tensor decomposition of 2x2 * 2x5."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0


# --------------------------------------------------------------------------- #
# Verification (exact, float64)
# --------------------------------------------------------------------------- #
def _verify(U, V, W, n=2, m=2, p=5, tol=1e-7):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                computed = 0.0
                for r in range(rank):
                    computed += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(computed - 1.0) > tol:
                    return False
    return True


def _build_target_tensor(n=2, m=2, p=5):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


# --------------------------------------------------------------------------- #
# Known fallback: rank-18 (Strassen 2x2x2 twice + schoolbook 2x2x1)
# --------------------------------------------------------------------------- #
def _construct_rank18(n=2, m=2, p=5):
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0 * p + col0][r] = 1; V[1 * p + col1][r] = 1
        W[col0 * n + 0][r] = 1; W[col1 * n + 1][r] = 1
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0 * p + col0][r] = 1
        W[col0 * n + 1][r] = 1; W[col1 * n + 1][r] = -1
        r = base + 2
        U[0][r] = 1
        V[0 * p + col1][r] = 1; V[1 * p + col1][r] = -1
        W[col1 * n + 0][r] = 1; W[col1 * n + 1][r] = 1
        r = base + 3
        U[3][r] = 1
        V[1 * p + col0][r] = 1; V[0 * p + col0][r] = -1
        W[col0 * n + 0][r] = 1; W[col0 * n + 1][r] = 1
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1 * p + col1][r] = 1
        W[col0 * n + 0][r] = -1; W[col1 * n + 0][r] = 1
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0 * p + col0][r] = 1; V[0 * p + col1][r] = 1
        W[col1 * n + 1][r] = 1
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1 * p + col0][r] = 1; V[1 * p + col1][r] = 1
        W[col0 * n + 0][r] = 1

    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    r = 14
    U[0][r] = 1; V[0 * p + 4][r] = 1; W[4 * n + 0][r] = 1
    r = 15
    U[1][r] = 1; V[1 * p + 4][r] = 1; W[4 * n + 0][r] = 1
    r = 16
    U[2][r] = 1; V[0 * p + 4][r] = 1; W[4 * n + 1][r] = 1
    r = 17
    U[3][r] = 1; V[1 * p + 4][r] = 1; W[4 * n + 1][r] = 1
    return {"U": U, "V": V, "W": W}


# --------------------------------------------------------------------------- #
# ALS refinement (with optional structured support)
# --------------------------------------------------------------------------- #
def _als_refine(U, V, W, T, R, n, m, p, maskU=None, maskV=None, maskW=None,
                max_iter=150, tol=1e-13, time_budget=None):
    start = time.time()
    prev_err = np.inf
    for it in range(max_iter):
        if time_budget is not None and time.time() - start > time_budget:
            break

        # ---- solve U ----
        VvW = np.einsum('vr,wr->vwr', V, W)
        VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
        for u in range(n * m):
            if maskU is not None and maskU[u] is not None:
                cols = maskU[u]
                if len(cols) == 0:
                    U[u, :] = 0.0
                    continue
                A = VvW_flat[:, cols]
                b = T[u].reshape(-1)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                U[u, :] = 0.0
                U[u, cols] = sol
            else:
                b = T[u].reshape(-1)
                sol, *_ = np.linalg.lstsq(VvW_flat, b, rcond=None)
                U[u, :] = sol

        # ---- solve V ----
        Uw = np.einsum('ur,wr->uwr', U, W)
        Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
        for v in range(m * p):
            if maskV is not None and maskV[v] is not None:
                cols = maskV[v]
                if len(cols) == 0:
                    V[v, :] = 0.0
                    continue
                A = Uw_flat[:, cols]
                b = T[:, v, :].reshape(-1)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                V[v, :] = 0.0
                V[v, cols] = sol
            else:
                b = T[:, v, :].reshape(-1)
                sol, *_ = np.linalg.lstsq(Uw_flat, b, rcond=None)
                V[v, :] = sol

        # ---- solve W ----
        UV = np.einsum('ur,vr->uvr', U, V)
        UV_flat = UV.reshape(-1, R)
        for w in range(p * n):
            if maskW is not None and maskW[w] is not None:
                cols = maskW[w]
                if len(cols) == 0:
                    W[w, :] = 0.0
                    continue
                A = UV_flat[:, cols]
                b = T[:, :, w].reshape(-1)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                W[w, :] = 0.0
                W[w, cols] = sol
            else:
                b = T[:, :, w].reshape(-1)
                sol, *_ = np.linalg.lstsq(UV_flat, b, rcond=None)
                W[w, :] = sol

        result = np.einsum('ur,vr,wr->uvw', U, V, W)
        err = np.linalg.norm(result - T)
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    return U, V, W, err


# --------------------------------------------------------------------------- #
# Rational recovery
# --------------------------------------------------------------------------- #
def _rationalize(U, V, W, T, R, n, m, p):
    result = np.einsum('ur,vr,wr->uvw', U, V, W)
    if not np.allclose(result, T, atol=1e-4):
        return None

    for denom in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                  16384, 32768, 65536, 131072, 262144, 524288]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        result_s = np.einsum('ur,vr,wr->uvw', U_s, V_s, W_s)
        if np.allclose(result_s, T, atol=1e-7):
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]
            for mat in (U_list, V_list, W_list):
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-9:
                            mat[i][j] = int(round(mat[i][j]))
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}

    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    return None


# --------------------------------------------------------------------------- #
# Structured random initializations
# --------------------------------------------------------------------------- #
def _structured_init(n, m, p, R, rng):
    """Each rank-1 term picks one (i,j) and one k, matching the tensor support."""
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    for r in range(R):
        i = rng.randint(0, n - 1)
        j = rng.randint(0, m - 1)
        k = rng.randint(0, p - 1)
        U[i * m + j, r] = rng.uniform(-1.0, 1.0)
        V[j * p + k, r] = rng.uniform(-1.0, 1.0)
        W[k * n + i, r] = rng.uniform(-1.0, 1.0)
    return U, V, W


def _structured_init2(n, m, p, R, rng):
    """Structured support with small off-support noise to help ALS escape."""
    U, V, W = _structured_init(n, m, p, R, rng)
    noise = 0.05
    U += noise * rng.standard_normal(U.shape)
    V += noise * rng.standard_normal(V.shape)
    W += noise * rng.standard_normal(W.shape)
    return U, V, W


def _random_init(n, m, p, R, rng):
    U = rng.standard_normal((n * m, R))
    V = rng.standard_normal((m * p, R))
    W = rng.standard_normal((p * n, R))
    return U, V, W


def _structured_masks(n, m, p, R, rng):
    """Per-row allowed rank-1 indices following the tensor support."""
    maskU = [None] * (n * m)
    maskV = [None] * (m * p)
    maskW = [None] * (p * n)
    for u in range(n * m):
        i, j = divmod(u, m)
        maskU[u] = [r for r in range(R) if rng.random() < 0.5]
    for v in range(m * p):
        j, k = divmod(v, p)
        maskV[v] = [r for r in range(R) if rng.random() < 0.5]
    for w in range(p * n):
        k, i = divmod(w, n)
        maskW[w] = [r for r in range(R) if rng.random() < 0.5]
    return maskU, maskV, maskW


# --------------------------------------------------------------------------- #
# Search for a target rank
# --------------------------------------------------------------------------- #
def _try_rank(n, m, p, R, T, timeout, num_trials):
    start = time.time()
    for trial in range(num_trials):
        if time.time() - start > timeout - 1.0:
            break
        rng = random.Random((trial * 7919 + 13) % (2 ** 31))
        np_rng = np.random.default_rng((trial * 104729 + 17) % (2 ** 32))

        if trial % 3 == 0:
            U, V, W = _structured_init(n, m, p, R, rng)
            mU = mV = mW = None
        elif trial % 3 == 1:
            U, V, W = _structured_init2(n, m, p, R, rng)
            mU = mV = mW = None
        else:
            U, V, W = _random_init(n, m, p, R, np_rng)
            mU, mV, mW = _structured_masks(n, m, p, R, rng)

        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p,
                                   maskU=mU, maskV=mV, maskW=mW,
                                   max_iter=120, tol=1e-13,
                                   time_budget=max(1.0, timeout - (time.time() - start) - 0.5))
        if err < 1e-3:
            result = _rationalize(U, V, W, T, R, n, m, p)
            if result is not None:
                return result
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT

    # Guaranteed fallback
    result18 = _construct_rank18(n, m, p)
    if not _verify(result18['U'], result18['V'], result18['W'], n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    T = _build_target_tensor(n, m, p)

    # Try rank 16, then 15
    for R in (16, 15):
        if time.time() > deadline - 8.0:
            break
        remaining = deadline - time.time()
        if remaining < 4.0:
            break
        timeout = min(remaining - 3.0, 40.0)
        num_trials = 60
        result = _try_rank(n, m, p, R, T, timeout, num_trials)
        if result is not None:
            return result

    return result18
