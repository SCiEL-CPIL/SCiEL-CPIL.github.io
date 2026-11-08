"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    deadline = time.time() + 130

    ranks_to_try = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    time_per_rank = {24: 12, 25: 10, 26: 10, 27: 8, 28: 7, 29: 6, 30: 5, 31: 5, 32: 4, 33: 4}

    for target_rank in ranks_to_try:
        if time.time() >= deadline - 10:
            break
        tl = time_per_rank.get(target_rank, 5.0)
        tl = min(tl, deadline - time.time() - 10)
        if tl < 3:
            break
        res = attempt_lower_rank(target_rank, time_limit=tl)
        if res is not None and valid_res(res, 3, 3, 4):
            return res

    return construct_r34()

def attempt_lower_rank(target_rank, time_limit=10.0):
    n, m, p = 3, 3, 4
    R = target_rank
    start_time = time.time()

    indices_a = []
    indices_b = []
    indices_c = []
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                indices_a.append(a)
                indices_b.append(b)
                indices_c.append(c)

    indices_a = np.array(indices_a)
    indices_b = np.array(indices_b)
    indices_c = np.array(indices_c)

    def objective(params):
        U = params[0:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        vals = np.sum(U[indices_a, :] * V[indices_b, :] * W[indices_c, :], axis=1)
        error = np.sum((vals - 1.0)**2)
        return error

    from scipy.optimize import minimize

    best_error = float('inf')
    num_restarts = min(10, max(2, int(time_limit / 1.0)))

    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 1:
            break

        np.random.seed(1000 + restart * 7919)

        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))

        if R >= 26 and restart < num_restarts // 2:
            # Structured: 23 terms for 3x3x3 + 3 for 4th col
            for r in range(23):
                u_idx = np.random.choice(n*m, size=np.random.randint(2, 4), replace=False)
                v_idx = np.random.choice(m*p, size=np.random.randint(2, 4), replace=False)
                w_idx = np.random.choice(3*n, size=np.random.randint(2, 4), replace=False)
                U[u_idx, r] = np.random.choice([-1, 1], size=len(u_idx))
                V[v_idx, r] = np.random.choice([-1, 1], size=len(v_idx))
                W[w_idx, r] = np.random.choice([-1, 1], size=len(w_idx))
            for i in range(3):
                r = 23 + i
                U[i*m, r] = 1.0
                V[:, r] = 1.0
                W[3*n + i, r] = 1.0
        elif restart < num_restarts // 2:
            # All random sparse
            for r in range(R):
                u_idx = np.random.choice(n*m, size=np.random.randint(2, 4), replace=False)
                v_idx = np.random.choice(m*p, size=np.random.randint(2, 4), replace=False)
                w_idx = np.random.choice(p*n, size=np.random.randint(2, 4), replace=False)
                U[u_idx, r] = np.random.choice([-1, 1], size=len(u_idx))
                V[v_idx, r] = np.random.choice([-1, 1], size=len(v_idx))
                W[w_idx, r] = np.random.choice([-1, 1], size=len(w_idx))
        else:
            # Dense random
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5

        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

        try:
            res = minimize(objective, params, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30})

            if res.fun < best_error:
                best_error = res.fun

                if res.fun < 1e-8:
                    U = res.x[0:n*m*R].reshape(n*m, R)
                    V = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W = res.x[n*m*R + m*p*R:].reshape(p*n, R)

                    for denom in [2, 4, 8, 16]:
                        U_rounded = np.round(U * denom) / denom
                        V_rounded = np.round(V * denom) / denom
                        W_rounded = np.round(W * denom) / denom
                        vals = np.sum(U_rounded[indices_a, :] * V_rounded[indices_b, :] * W_rounded[indices_c, :], axis=1)
                        if np.allclose(vals, 1.0, atol=1e-9):
                            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}

                    res2 = minimize(objective, res.x, method='L-BFGS-B',
                                   options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12})
                    if res2.fun < 1e-10:
                        U = res2.x[0:n*m*R].reshape(n*m, R)
                        V = res2.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                        W = res2.x[n*m*R + m*p*R:].reshape(p*n, R)
                        U_rounded = np.round(U * 16) / 16
                        V_rounded = np.round(V * 16) / 16
                        W_rounded = np.round(W * 16) / 16
                        vals = np.sum(U_rounded[indices_a, :] * V_rounded[indices_b, :] * W_rounded[indices_c, :], axis=1)
                        if np.allclose(vals, 1.0, atol=1e-9):
                            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        except Exception:
            continue

    return None

def valid_res(res, n, m, p):
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
        
    R = U.shape[1]
    if V.shape[1] != R or W.shape[1] != R:
        return False
        
    indices_a = []
    indices_b = []
    indices_c = []
    for i in range(n):
        for k in range(m):
            for j in range(p):
                indices_a.append(i * m + k)
                indices_b.append(k * p + j)
                indices_c.append(j * n + i)
                
    indices_a = np.array(indices_a)
    indices_b = np.array(indices_b)
    indices_c = np.array(indices_c)
    
    vals = np.sum(U[indices_a, :] * V[indices_b, :] * W[indices_c, :], axis=1)
    return np.allclose(vals, 1.0, atol=1e-9)

def construct_r34():
    """Fallback: naive rank-36 decomposition (guaranteed valid)."""
    n, m, p = 3, 3, 4
    R = n * m * p  # 36
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))

    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
