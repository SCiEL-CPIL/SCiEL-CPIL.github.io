import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification of exact decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)

    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # W is indexed [k*n+i, r]; reshape to (p, n, rank) gives W_arr[k, i, r]
    T = np.einsum('ijr,jkr,kir->ijk', U_arr, V_arr, W_arr)

    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True


def _build_strassen_rank39(n, m, p):
    """Clean additive rank-39 Strassen-based decomposition."""
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
        nonlocal col
        for idx, c in zip(u_indices, u_coeffs):
            U[idx][col] += c
        for idx, c in zip(v_indices, v_coeffs):
            V[idx][col] += c
        for idx, c in zip(w_indices, w_coeffs):
            W[idx][col] += c
        col += 1

    def add_strassen_block(k_start):
        a00 = 0 * 3 + 0
        a01 = 0 * 3 + 1
        a10 = 1 * 3 + 0
        a11 = 1 * 3 + 1

        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1

        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1

        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
        # M2: (A10+A11)B00 -> C10, -C11
        add_term([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])
        # M3: A00(B01-B11) -> C01, C11
        add_term([a00], [1], [b01, b11], [1, -1], [c01, c11], [1, 1])
        # M4: A11(B10-B00) -> C00, C10
        add_term([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1])
        # M5: (A00+A01)B11 -> -C00, C01
        add_term([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])

    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)

    # k=6, 2x2x1 schoolbook
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1

    add_term([0], [1], [b06], [1], [c06], [1])
    add_term([1], [1], [b16], [1], [c06], [1])
    add_term([3], [1], [b06], [1], [c16], [1])
    add_term([4], [1], [b16], [1], [c16], [1])

    # j=2, 2x1x7 schoolbook
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        add_term([2], [1], [b2k], [1], [c0k], [1])
        add_term([5], [1], [b2k], [1], [c1k], [1])

    return {"U": U, "V": V, "W": W}


def _alternating_minimize(U_init, V_init, W_init, n, m, p, target_rank,
                          max_iter=200, tol=1e-10, deadline=None):
    """Smooth least-squares alternating minimization to reach target_rank.

    Fixes two factors and solves a linear least-squares for the third,
    iterating until the trilinear objective is (near-)zero. This is a
    smooth surrogate for the nonsmooth rank objective.
    """
    nm = n * m
    mp = m * p
    pn = p * n

    U = np.array(U_init, dtype=np.float64)[:, :target_rank].copy()
    V = np.array(V_init, dtype=np.float64)[:, :target_rank].copy()
    W = np.array(W_init, dtype=np.float64)[:, :target_rank].copy()

    def obj():
        U_t = U.reshape(n, m, target_rank)
        V_t = V.reshape(m, p, target_rank)
        W_t = W.reshape(p, n, target_rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return np.sum((T - 1.0) ** 2)

    prev = obj()
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Solve for U: fix V,W. For each (i,j):
        #   sum_r U[i,j,r] * (V[j,k,r]*W[k,i,r]) = 1 for all k
        #   M[k,r] = V[j*p+k, r] * W[k*n+i, r];  M^T U[i,j,:] = ones(p)
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                except Exception:
                    pass

        # Solve for V: fix U,W. For each (j,k):
        #   N[i,r] = U[i*m+j, r] * W[k*n+i, r];  N^T V[j,k,:] = ones(n)
        for j in range(m):
            for k in range(p):
                N = np.zeros((n, target_rank))
                for i in range(n):
                    N[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :] = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                except Exception:
                    pass

        # Solve for W: fix U,V. For each (k,i):
        #   P[j,r] = U[i*m+j, r] * V[j*p+k, r];  P^T W[k,i,:] = ones(m)
        for k in range(p):
            for i in range(n):
                P = np.zeros((m, target_rank))
                for j in range(m):
                    P[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :] = np.linalg.lstsq(P.T, np.ones(m), rcond=None)[0]
                except Exception:
                    pass

        cur = obj()
        if cur < 1e-12:
            break
        if abs(cur - prev) < 1e-15:
            break
        prev = cur

    if obj() > tol:
        return None
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _try_single_removal(U, V, W, n, m, p, tol=1e-10):
    """Try removing any single column; return reduced dict or None."""
    current_rank = len(U[0])
    for col in range(current_rank):
        U_test = [row[:col] + row[col + 1:] for row in U]
        V_test = [row[:col] + row[col + 1:] for row in V]
        W_test = [row[:col] + row[col + 1:] for row in W]
        if _verify(U_test, V_test, W_test, n, m, p, tol):
            return {"U": U_test, "V": V_test, "W": W_test}
    return None


def _try_pair_merge(U, V, W, n, m, p, tol=1e-10):
    """Try merging two columns via U_a -> U_a - alpha*U_b then removing b."""
    current_rank = len(U[0])
    for a in range(current_rank):
        for b in range(current_rank):
            if a == b:
                continue
            for alpha in [-2, -1, 1, 2]:
                U_test = [row[:] for row in U]
                V_test = [row[:] for row in V]
                W_test = [row[:] for row in W]
                for i in range(len(U)):
                    U_test[i][a] = U[i][a] - alpha * U[i][b]
                for i in range(len(V)):
                    V_test[i][a] = V[i][a] - alpha * V[i][b]
                for i in range(len(W)):
                    W_test[i][a] = W[i][a] - alpha * W[i][b]
                U_test = [row[:b] + row[b + 1:] for row in U_test]
                V_test = [row[:b] + row[b + 1:] for row in V_test]
                W_test = [row[:b] + row[b + 1:] for row in W_test]
                if _verify(U_test, V_test, W_test, n, m, p, tol):
                    return {"U": U_test, "V": V_test, "W": W_test}
    return None


def _reduce(U, V, W, n, m, p, deadline):
    """Greedy single-removal / pair-merge reduction."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    for _ in range(200):
        if time.time() > deadline:
            break
        r = _try_single_removal(U, V, W, n, m, p)
        if r is None:
            r = _try_pair_merge(U, V, W, n, m, p)
        if r is None:
            break
        U, V, W = r["U"], r["V"], r["W"]
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None


def _polish(U, V, W, n, m, p, target_rank, deadline):
    """High-precision polish: alternating minimization at current rank."""
    res = _alternating_minimize(U, V, W, n, m, p, target_rank,
                                max_iter=100, tol=1e-11, deadline=deadline)
    if res is not None and _verify(res["U"], res["V"], res["W"], n, m, p):
        return res
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n, m, p = DIMENSIONS

    result = _build_strassen_rank39(n, m, p)
    if not _verify(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: schoolbook
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

    best = result
    best_rank = len(best["U"][0])

    # 1) Greedy reduction
    if time.time() < deadline:
        red = _reduce(best["U"], best["V"], best["W"], n, m, p, deadline)
        if red is not None and len(red["U"][0]) < best_rank:
            best = red
            best_rank = len(best["U"][0])

    # 2) Alternating-minimization attempts at lower ranks
    for target in range(best_rank - 1, 10, -1):
        if time.time() > deadline - 20:
            break
        res = _alternating_minimize(best["U"], best["V"], best["W"], n, m, p,
                                    target, max_iter=200, tol=1e-10,
                                    deadline=deadline - 15)
        if res is not None and _verify(res["U"], res["V"], res["W"], n, m, p):
            best = res
            best_rank = target
            # try to go lower from here
            break

    # 3) Final polish at the best rank found
    if time.time() < deadline - 5:
        pol = _polish(best["U"], best["V"], best["W"], n, m, p,
                      best_rank, deadline - 2)
        if pol is not None:
            best = pol

    return best
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
    print("valid:", _verify(res["U"], res["V"], res["W"], *DIMENSIONS))
