"""Rewritten program: rank-minimal exact tensor decomposition for 2x3 x 3x4 matrix mult."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    # Precompute index maps
    u_idx = np.array([i * m + j for (i, j, k) in indices])
    v_idx = np.array([j * p + k for (i, j, k) in indices])
    w_idx = np.array([k * n + i for (i, j, k) in indices])

    def residual(params, rank):
        U = params[:nm * rank].reshape(nm, rank)
        V = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W = params[nm * rank + mp * rank:].reshape(pn, rank)
        # For each (i,j,k): dot(U[u], V[v]*W[w]) = 1
        # = sum_r U[u,r] V[v,r] W[w,r]
        prod = V[v_idx, :] * W[w_idx, :]  # (24, rank)
        res = np.einsum('ir,ir->i', U[u_idx, :], prod) - 1.0
        return res

    def verify(U, V, W):
        for (i, j, k) in indices:
            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
            if abs(val - 1.0) > 1e-8:
                return False
        return True

    def snap_half(arr):
        # Round to nearest half-integer if close
        r = np.round(arr * 2) / 2.0
        if np.max(np.abs(arr - r)) < 1e-5:
            return r
        return arr

    def to_list(arr):
        lst = []
        for row in arr:
            row_lst = []
            for val in row:
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-6:
                    row_lst.append(rounded)
                else:
                    row_lst.append(float(val))
            lst.append(row_lst)
        return lst

    best_rank = None
    best_solution = None

    # Try ranks from low to high
    for rank in range(12, 25):
        if time.time() > deadline - 15:
            break

        remaining = deadline - 15 - time.time()
        max_attempts = max(2, int(remaining / 2.5))

        for attempt in range(max_attempts):
            if time.time() > deadline - 8:
                break

            np.random.seed(rank * 100000 + attempt * 131 + 7)
            scale = 1.0 / (rank ** 0.25)
            U = np.random.randn(nm, rank) * scale
            V = np.random.randn(mp, rank) * scale
            W = np.random.randn(pn, rank) * scale

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            try:
                result = least_squares(residual, x0, args=(rank,), method='lm', max_nfev=500)
                if result.cost < 1e-10:
                    U_sol = result.x[:nm * rank].reshape(nm, rank)
                    V_sol = result.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                    W_sol = result.x[nm * rank + mp * rank:].reshape(pn, rank)

                    if verify(U_sol, V_sol, W_sol):
                        # Polish: snap to half-integers and re-verify
                        U_s = snap_half(U_sol)
                        V_s = snap_half(V_sol)
                        W_s = snap_half(W_sol)
                        if verify(U_s, V_s, W_s):
                            U_sol, V_sol, W_sol = U_s, V_s, W_s
                        best_rank = rank
                        best_solution = (U_sol, V_sol, W_sol)
                        break
            except Exception:
                continue

        if best_rank is not None:
            break

    if best_rank is not None:
        U_sol, V_sol, W_sol = best_solution
        return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}

    # Fallback: naive decomposition (rank 24)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    return {"U": U, "V": V, "W": W}
