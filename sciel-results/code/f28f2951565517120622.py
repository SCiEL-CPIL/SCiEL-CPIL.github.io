import numpy as np
import time

def construct():
    n, m, p = 3, 3, 5
    deadline = time.time() + 140.0

    def verify_decomposition(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
        return U, V, W, rank

    best_U, best_V, best_W, best_rank = None, None, None, 45

    from scipy.optimize import least_squares
    
    def residuals(params, R):
        U_flat = params[:9 * R].reshape(9, R)
        V_flat = params[9 * R:9 * R + 15 * R].reshape(15, R)
        W_flat = params[9 * R + 15 * R:].reshape(15, R)
        
        res = np.zeros(45)
        idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(5):
                    u = U_flat[i * 3 + j]
                    v = V_flat[j * 5 + k]
                    w = W_flat[k * 3 + i]
                    res[idx] = np.dot(u, v * w) - 1.0
                    idx += 1
        return res

    # Try rank 40
    target_ranks = [40, 41, 42, 43, 44]
    
    for R in target_ranks:
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try a few restarts for this rank
        for restart in range(3):
            if time.time() > deadline - 10:
                break
                
            rng = np.random.default_rng(42 + R * 100 + restart * 13)
            U_init = rng.normal(size=(9, R)) * 0.3
            V_init = rng.normal(size=(15, R)) * 0.3
            W_init = rng.normal(size=(15, R)) * 0.3
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(
                    lambda p: residuals(p, R),
                    x0,
                    method='lm',
                    x_scale='jac',
                    max_nfev=5000,
                    ftol=1e-15,
                    xtol=1e-15,
                    gtol=1e-15
                )
                
                if res.cost < 1e-12:
                    U_sol = res.x[:9 * R].reshape(9, R)
                    V_sol = res.x[9 * R:9 * R + 15 * R].reshape(15, R)
                    W_sol = res.x[9 * R + 15 * R:].reshape(15, R)
                    
                    # Try to round to rational numbers
                    for scale in [1, 2, 4, 10]:
                        U_r = np.round(U_sol * scale) / scale
                        V_r = np.round(V_sol * scale) / scale
                        W_r = np.round(W_sol * scale) / scale
                        
                        U_list = U_r.tolist()
                        V_list = V_r.tolist()
                        W_list = W_r.tolist()
                        
                        if verify_decomposition(U_list, V_list, W_list, 3, 3, 5):
                            best_U, best_V, best_W, best_rank = U_list, V_list, W_list, R
                            found = True
                            break
                    if found:
                        break
            except Exception:
                pass
        
        if found:
            break

    if best_U is None:
        best_U, best_V, best_W, best_rank = make_schoolbook(3, 3, 5)
    
    return {"U": best_U, "V": best_V, "W": best_W}
