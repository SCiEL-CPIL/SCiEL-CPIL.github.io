"""Rewritten: focused rank search with structured init and fast ALS for 3x4x4 matmul."""
import time
import numpy as np
from scipy import linalg

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_TIMEOUT = 140


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


def _tensor_err(U, V, W, n, m, p):
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    return float(np.max(np.abs(T - 1.0)))


def _als_step(U, V, W, n, m, p):
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)

    U_new = np.zeros_like(U)
    V_new = np.zeros_like(V)
    W_new = np.zeros_like(W)

    # Solve U rows
    for i in range(n):
        for j in range(m):
            A_mat = V3[j] * W3[:, i]  # (p, rank)
            try:
                sol, *_ = linalg.lstsq(A_mat, np.ones(p), rcond=None)
            except Exception:
                sol = U[i * m + j]
            U_new[i * m + j] = sol

    # Solve V rows
    for j in range(m):
        for k in range(p):
            A_mat = U3[:, j] * W3[k]  # (n, rank)
            try:
                sol, *_ = linalg.lstsq(A_mat, np.ones(n), rcond=None)
            except Exception:
                sol = V[j * p + k]
            V_new[j * p + k] = sol

    # Solve W rows
    for k in range(p):
        for i in range(n):
            A_mat = U3[:, :, None] * V3[:, k, None]  # this is wrong shape
            # A_mat[i,j] should be U[i,j,:]*V[j,k,:]
            A_mat = np.zeros((m, rank))
            for jj in range(m):
                A_mat[jj] = U3[i, jj] * V3[jj, k]
            try:
                sol, *_ = linalg.lstsq(A_mat, np.ones(m), rcond=None)
            except Exception:
                sol = W[k * n + i]
            W_new[k * n + i] = sol

    return U_new, V_new, W_new


def _als(U, V, W, n, m, p, max_iter=300, tol=1e-12):
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = _tensor_err(U, V, W, n, m, p)
    for _ in range(max_iter):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        U, V, W = _als_step(U, V, W, n, m, p)
        err = _tensor_err(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if best_err < tol:
            break
    return best_U, best_V, best_W, best_err


def _structured_init(rank, n, m, p, seed):
    """Initialize with structure: spread ones across the rank dimension."""
    np.random.seed(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Assign each (i,j,k) to a rank index, with possible sharing
    # Use a hash-like assignment
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = (i * m * p + j * p + k) % rank
                U[i * m + j, r] += 1.0
                V[j * p + k, r] += 1.0
                W[k * n + i, r] += 1.0

    # Add small random perturbation to break symmetry
    U += 0.01 * np.random.randn(n * m, rank)
    V += 0.01 * np.random.randn(m * p, rank)
    W += 0.01 * np.random.randn(p * n, rank)
    return U, V, W


def _try_rank(rank, n, m, p, seed):
    U0, V0, W0 = _structured_init(rank, n, m, p, seed)
    U_sol, V_sol, W_sol, err = _als(U0, V0, W0, n, m, p, max_iter=200)
    if err < 1e-4 and time.time() - _START_TIME < _TIMEOUT - 10:
        U_p, V_p, W_p, err_p = _als(U_sol, V_sol, W_sol, n, m, p, max_iter=100, tol=1e-13)
        if err_p < err:
            return U_p, V_p, W_p, err_p
    return U_sol, V_sol, W_sol, err


def construct():
    n, m, p = DIMENSIONS
    max_rank = n * m * p

    U_fb, V_fb, W_fb = _schoolbook()
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = max_rank

    # Known: rank of 3x4x4 matmul is 33. Search from 30 to 47.
    for target_rank in range(30, max_rank):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            break

        found = False
        for restart in range(5):
            if time.time() - _START_TIME > _TIMEOUT - 3:
                break
            U_t, V_t, W_t, err = _try_rank(target_rank, n, m, p, seed=target_rank * 1000 + restart)
            if err < 1e-9:
                best_U, best_V, best_W = U_t, V_t, W_t
                best_rank = target_rank
                found = True
                break

        if found:
            continue  # try lower next

    U_list = [[float(x) for x in row] for row in best_U.tolist()]
    V_list = [[float(x) for x in row] for row in best_V.tolist()]
    W_list = [[float(x) for x in row] for row in best_W.tolist()]

    return {"U": U_list, "V": V_list, "W": W_list}
