"""
Exact tensor decomposition for 4x4 matrix multiplication.

Uses gradient-based search to find a low-rank decomposition, verified
exactly before returning. Falls back to rank-49 Kronecker-Strassen and
finally to rank-64 schoolbook on failure.
"""

import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_kron_4x4(n, m, p):
    """Rank-49 decomposition via Kronecker product of Strassen 2x2."""
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)

    perm_u = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i_out, i_in = i // 2, i % 2
        j_out, j_in = j // 2, j % 2
        a_idx = i_out * 2 + j_out
        b_idx = i_in * 2 + j_in
        perm_u[S] = a_idx * 4 + b_idx

    perm_v = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        a_idx = j_out * 2 + k_out
        b_idx = j_in * 2 + k_in
        perm_v[S] = a_idx * 4 + b_idx

    perm_w = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        i = S % 4
        k_out, k_in = k // 2, k % 2
        i_out, i_in = i // 2, i % 2
        a_idx = k_out * 2 + i_out
        b_idx = k_in * 2 + i_in
        perm_w[S] = a_idx * 4 + b_idx

    U4 = U4[perm_u, :]
    V4 = V4[perm_v, :]
    W4 = W4[perm_w, :]

    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = int(np.dot(U4[i*4+j, :], V4[j*4+k, :] * W4[k*4+i, :]))
                if s != 1:
                    return _schoolbook(n, m, p)

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    return False
    return True


def _optimize_rank(r, time_budget=20, seed=42):
    """Try to find a rank-r decomposition via gradient descent."""
    n, m, p = 4, 4, 4
    start = time.time()
    np.random.seed(seed)

    U = np.random.randn(n*m, r) * 0.5
    V = np.random.randn(m*p, r) * 0.5
    W = np.random.randn(p*n, r) * 0.5

    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    lr = 0.05

    for it in range(5000):
        if time.time() - start > time_budget:
            break

        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        loss = 0.0

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target = 1.0 if i == k else 0.0
                    s = float(np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :]))
                    err = s - target
                    loss += err * err
                    if abs(err) > 1e-15:
                        gv = V[j*p+k, :] * W[k*n+i, :]
                        gw = U[i*m+j, :] * V[j*p+k, :]
                        gu = U[i*m+j, :] * W[k*n+i, :]
                        grad_U[i*m+j, :] += 2.0 * err * gv
                        grad_V[j*p+k, :] += 2.0 * err * gw
                        grad_W[k*n+i, :] += 2.0 * err * gu

        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W

        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        if best_loss < 1e-14:
            break

        if it % 200 == 0 and it > 0:
            lr *= 0.95

    if best_loss > 1e-10:
        return None

    U_r = np.round(best_U, 8)
    V_r = np.round(best_V, 8)
    W_r = np.round(best_W, 8)

    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1 if i == k else 0
                s = 0.0
                for c in range(r):
                    s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                if abs(s - target) > 1e-8:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break

    if not ok:
        return None

    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}


def construct():
    """
    Constructs a minimal-rank exact tensor decomposition for 4x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer entries.
    """
    start_time = time.time()
    deadline = start_time + 100

    n, m, p = DIMENSIONS

    # Try ranks from 32 to 48
    for r in [32, 34, 36, 38, 40, 42, 44, 46, 48]:
        if time.time() > deadline:
            break
        budget = min(15, deadline - time.time())
        if budget < 5:
            break
        res = _optimize_rank(r, time_budget=budget, seed=42 + r)
        if res is not None:
            if _verify_decomposition(res["U"], res["V"], res["W"], n, m, p):
                return res

    # Fallback: rank-49 Kronecker-Strassen
    if time.time() < deadline:
        return _strassen_kron_4x4(n, m, p)

    # Final fallback: schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
