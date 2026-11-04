import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p):
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    nm, mp, pn = n * m, m * p, p * n

    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                val = np.sum(U_np[u, :] * V_np[v, :] * W_np[w, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False

    for u in range(nm):
        for v in range(mp):
            for w in range(pn):
                is_nz = False
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if u == i * m + j and v == j * p + k and w == k * n + i:
                                is_nz = True
                if not is_nz:
                    val = np.sum(U_np[u, :] * V_np[v, :] * W_np[w, :])
                    if not np.isclose(val, 0.0, atol=1e-9):
                        return False
    return True

def _compute_residual(U, V, W, nz_indices, n, m, p):
    nm, mp, pn = n * m, m * p, p * n
    residual = 0.0
    for (u, v, w) in nz_indices:
        val = np.sum(U[u, :] * V[v, :] * W[w, :])
        residual += (val - 1.0) ** 2
    # Sample some zero entries
    for u in range(0, nm, 2):
        for v in range(0, mp, 2):
            for w in range(0, pn, 2):
                is_nz = any(u == uu and v == vv and w == ww for (uu, vv, ww) in nz_indices)
                if not is_nz:
                    val = np.sum(U[u, :] * V[v, :] * W[w, :])
                    residual += val ** 2
    return residual

def _gradient_step(U, V, W, nz_indices, n, m, p, lr):
    nm, mp, pn = n * m, m * p, p * n
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)

    for (u, v, w) in nz_indices:
        val = np.sum(U[u, :] * V[v, :] * W[w, :])
        err = val - 1.0
        grad_U[u, :] += err * V[v, :] * W[w, :]
        grad_V[v, :] += err * U[u, :] * W[w, :]
        grad_W[w, :] += err * U[u, :] * V[v, :]

    # Add zero-entry penalties
    for u in range(0, nm, 3):
        for v in range(0, mp, 3):
            for w in range(0, pn, 3):
                is_nz = any(u == uu and v == vv and w == ww for (uu, vv, ww) in nz_indices)
                if not is_nz:
                    val = np.sum(U[u, :] * V[v, :] * W[w, :])
                    grad_U[u, :] += val * V[v, :] * W[w, :]
                    grad_V[v, :] += val * U[u, :] * W[w, :]
                    grad_W[w, :] += val * U[u, :] * V[v, :]

    U -= lr * grad_U
    V -= lr * grad_V
    W -= lr * grad_W
    return U, V, W

def _try_find_decomposition(n, m, p, target_rank, time_limit):
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n

    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_indices.append((i * m + j, j * p + k, k * n + i))

    for attempt in range(50):
        if time.time() > time_limit:
            break

        U = np.random.randn(nm, target_rank) * 0.3
        V = np.random.randn(mp, target_rank) * 0.3
        W = np.random.randn(pn, target_rank) * 0.3

        lr = 0.01
        best_residual = _compute_residual(U, V, W, nz_indices, n, m, p)

        for it in range(200):
            if time.time() > time_limit:
                break

            U, V, W = _gradient_step(U, V, W, nz_indices, n, m, p, lr)
            residual = _compute_residual(U, V, W, nz_indices, n, m, p)

            if residual < best_residual:
                best_residual = residual
                # Decay learning rate
                lr *= 0.98
            else:
                lr *= 0.95

            # Check for half-integer solution
            if it % 10 == 0:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                r_res = _compute_residual(U_r, V_r, W_r, nz_indices, n, m, p)
                if r_res < 1e-6:
                    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()

    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 140.0

    for r in range(27, min(rank_schoolbook, 36)):
        if time.time() > deadline - 10:
            break
        result = _try_find_decomposition(n, m, p, r, deadline - 5)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
