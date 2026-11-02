import numpy as np
import time
import sys
from scipy.optimize import least_squares, minimize

DIMENSIONS = [3, 3, 4]

def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p, tol=1e-8):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check diagonal terms (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]))
                if abs(s - 1.0) > tol:
                    return False
                # Check off-diagonal terms (should be 0)
                for j2 in range(m):
                    if j2 != j:
                        s_off = float(np.dot(U[i*m+j, :], V[j2*p+k, :] * W[k*n+i, :]))
                        if abs(s_off) > tol:
                            return False
    return True

def _check_and_round(U, V, W, tol=1e-6):
    """Check if U, V, W are close to half-integers and return rounded versions if so."""
    Uf = np.array(U, dtype=float).flatten()
    Vf = np.array(V, dtype=float).flatten()
    Wf = np.array(W, dtype=float).flatten()

    dist_U = np.abs(Uf - np.round(2 * Uf) / 2.0)
    dist_V = np.abs(Vf - np.round(2 * Vf) / 2.0)
    dist_W = np.abs(Wf - np.round(2 * Wf) / 2.0)

    if np.max(dist_U) < tol and np.max(dist_V) < tol and np.max(dist_W) < tol:
        U_r = np.round(2 * Uf) / 2.0
        V_r = np.round(2 * Vf) / 2.0
        W_r = np.round(2 * Wf) / 2.0

        n, m, p = DIMENSIONS
        U_3d = U_r.reshape(n*m, -1)
        V_3d = V_r.reshape(m*p, -1)
        W_3d = W_r.reshape(p*n, -1)
        rank = U_3d.shape[1]

        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(U_3d[i*m+j, :], V_3d[j*p+k, :] * W_3d[k*n+i, :]))
                    if abs(val - 1.0) > tol:
                        error = 1.0
                        break
                    for j2 in range(m):
                        if j2 != j:
                            val_off = float(np.dot(U_3d[i*m+j, :], V_3d[j2*p+k, :] * W_3d[k*n+i, :]))
                            if abs(val_off) > tol:
                                error = 1.0
                                break
                    if error: break
                if error: break
            if error: break
        if error == 0.0:
            return True, U_r.reshape(np.array(U).shape), V_r.reshape(np.array(V).shape), W_r.reshape(np.array(W).shape)
    return False, None, None, None

def _structured_seed(n, m, p, rank, seed=0):
    rng = np.random.RandomState(seed)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= rank:
                    break
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1

    for c in range(col, rank):
        U[:, c] = rng.normal(0, 0.05)
        V[:, c] = rng.normal(0, 0.05)
        W[:, c] = rng.normal(0, 0.05)

    return U, V, W

def _residual(x, n, m, p, rank):
    size_U = n * m * rank
    size_V = m * p * rank
    Uf = x[:size_U].reshape(n*m, rank)
    Vf = x[size_U:size_U+size_V].reshape(m*p, rank)
    Wf = x[size_U+size_V:].reshape(p*n, rank)

    res = []
    for i in range(n):
        for j in range(m):
            u_row = Uf[i*m+j, :]
            for k in range(p):
                w_row = Wf[k*n+i, :]
                for j2 in range(m):
                    v_row = Vf[j2*p+k, :]
                    val = float(np.dot(u_row, v_row * w_row))
                    target = 1.0 if j2 == j else 0.0
                    res.append(val - target)
    return np.array(res)

def _polish(U, V, W, n, m, p, rank, max_nfev=1000, tol=1e-10):
    size_U = n * m * rank
    size_V = m * p * rank
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    best_x = x0
    best_cost = np.inf
    for attempt in range(3):
        try:
            res = least_squares(_residual, x0, args=(n, m, p, rank),
                                method='trf', xtol=tol, ftol=tol, max_nfev=max_nfev)
            cost = res.cost
            if cost < best_cost:
                best_cost = cost
                best_x = res.x
            x0 = best_x
            if best_cost < 1e-14:
                break
        except Exception:
            break

    U_sol = best_x[:size_U].reshape(n*m, rank)
    V_sol = best_x[size_U:size_U+size_V].reshape(m*p, rank)
    W_sol = best_x[size_U+size_V:].reshape(p*n, rank)
    return U_sol, V_sol, W_sol, best_cost

def _try_find_decomposition(n, m, p, rank, time_budget=20.0):
    start = time.time()
    for seed in [0, 1, 2, 3, 4, 5]:
        if time.time() - start > time_budget:
            break
        U0, V0, W0 = _structured_seed(n, m, p, rank, seed=seed)
        U_sol, V_sol, W_sol, cost = _polish(U0, V0, W0, n, m, p, rank)
        if cost < 1e-8:
            valid, U_r, V_r, W_r = _check_and_round(U_sol, V_sol, W_sol)
            if valid:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            if _verify(U_sol, V_sol, W_sol, n, m, p, tol=1e-9):
                return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    ranks_to_try = [35, 34, 33, 32]

    for rank in ranks_to_try:
        if time.time() > deadline - 8:
            break
        remaining = max(2.0, min(15.0, deadline - time.time() - 8))
        result = _try_find_decomposition(n, m, p, rank, time_budget=remaining)
        if result:
            if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
                return result

    return best
