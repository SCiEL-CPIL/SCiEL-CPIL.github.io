import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    def check_solution(U, V, W, R):
        max_err = 0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0
                        for r in range(R):
                            s += W[k * n + i, r] * U[i * m + j1, r] * V[j2 * p + k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        err = abs(s - target)
                        if err > max_err:
                            max_err = err
        return max_err

    def round_to_half(x):
        return np.round(x * 2) / 2.0

    def polish_solution(U, V, W, R):
        # Flatten U, V, W
        U_flat = U.flatten()
        V_flat = V.flatten()
        W_flat = W.flatten()
        x0 = np.concatenate([U_flat, V_flat, W_flat])
        
        def residuals(x):
            U = x[:n*m*R].reshape(n*m, R)
            V = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W = x[n*m*R + m*p*R:].reshape(p*n, R)
            
            res = []
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            s = 0
                            for r in range(R):
                                s += W[k * n + i, r] * U[i * m + j1, r] * V[j2 * p + k, r]
                            target = 1.0 if j1 == j2 else 0.0
                            res.append(s - target)
            return np.array(res)
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=1000)
            U_pol = sol.x[:n*m*R].reshape(n*m, R)
            V_pol = sol.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W_pol = sol.x[n*m*R + m*p*R:].reshape(p*n, R)
            return U_pol, V_pol, W_pol
        except:
            return U, V, W

    def als_decompose(R, max_iters=10, tol=1e-10):
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None, None, None, 1.0
            
            # Update W
            new_W = np.zeros_like(W)
            for i in range(n):
                for k in range(p):
                    X = np.zeros((m * m, R))
                    for j1 in range(m):
                        for j2 in range(m):
                            X[j1 * m + j2, :] = U[i * m + j1, :] * V[j2 * p + k, :]
                    y = np.zeros(m * m)
                    for j in range(m):
                        y[j * m + j] = 1.0
                    try:
                        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        new_W[k * n + i, :] = w
                    except:
                        return None, None, None, 1.0
            W = new_W
            
            # Update U
            new_U = np.zeros_like(U)
            for i in range(n):
                for j1 in range(m):
                    X = np.zeros((p * m, R))
                    y = np.zeros(p * m)
                    for k in range(p):
                        for j2 in range(m):
                            X[k * m + j2, :] = W[k * n + i, :] * V[j2 * p + k, :]
                            y[k * m + j2] = 1.0 if j2 == j1 else 0.0
                    try:
                        u, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        new_U[i * m + j1, :] = u
                    except:
                        return None, None, None, 1.0
            U = new_U
            
            # Update V
            new_V = np.zeros_like(V)
            for k in range(p):
                for j2 in range(m):
                    X = np.zeros((n * m, R))
                    y = np.zeros(n * m)
                    for i in range(n):
                        for j1 in range(m):
                            X[i * m + j1, :] = W[k * n + i, :] * U[i * m + j1, :]
                            y[i * m + j1] = 1.0 if j1 == j2 else 0.0
                    try:
                        v, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        new_V[j2 * p + k, :] = v
                    except:
                        return None, None, None, 1.0
            V = new_V
            
            err = check_solution(U, V, W, R)
            if err < tol:
                # Polish
                U_pol, V_pol, W_pol = polish_solution(U, V, W, R)
                err_pol = check_solution(U_pol, V_pol, W_pol, R)
                if err_pol < err:
                    U, V, W = U_pol, V_pol, W_pol
                    err = err_pol
                return U, V, W, err
        
        # Final polish
        U_pol, V_pol, W_pol = polish_solution(U, V, W, R)
        err_pol = check_solution(U_pol, V_pol, W_pol, R)
        if err_pol < check_solution(U, V, W, R):
            U, V, W = U_pol, V_pol, W_pol
        return U, V, W, check_solution(U, V, W, R)

    best_rank = n * m * p
    best_U = np.array(schoolbook()["U"])
    best_V = np.array(schoolbook()["V"])
    best_W = np.array(schoolbook()["W"])

    # Try ranks from 104 down to 90
    for R in range(104, 89, -1):
        if time.time() > deadline:
            break
        for trial in range(3):
            if time.time() > deadline:
                break
            U, V, W, err = als_decompose(R)
            if U is None:
                continue
            if err < 1e-8:
                # Try rounding
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                err_r = check_solution(U_r, V_r, W_r, R)
                if err_r < 1e-8:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                # If rounding fails, check if original is close enough?
                # No, must be integer/half-integer.
                # If we can't round, we can't use it unless it's already integer/half-integer.
                # Check if U, V, W are already close to half-integers.
                if np.allclose(U, U_r) and np.allclose(V, V_r) and np.allclose(W, W_r):
                     return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    return schoolbook()
