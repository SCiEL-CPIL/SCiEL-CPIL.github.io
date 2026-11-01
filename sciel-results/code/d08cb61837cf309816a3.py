import numpy as np
import time
import scipy.optimize as so

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Precompute index mappings for ALS
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    # For updating U:
    # For each u (i, j), we have equations for k=0..p-1
    # B_u[k, r] = V[j*p+k, r] * W[k*n+i, r]
    U_indices = []
    for i in range(n):
        for j in range(m):
            u = i * m + j
            # Equations for each k
            eqs = []
            for k in range(p):
                v = j * p + k
                w = k * n + i
                eqs.append((v, w))
            U_indices.append((i, j, u, eqs))
            
    # For updating V:
    # For each v (j, k), we have equations for i=0..n-1
    # C_v[i, r] = U[i*m+j, r] * W[k*n+i, r]
    V_indices = []
    for j in range(m):
        for k in range(p):
            v = j * p + k
            eqs = []
            for i in range(n):
                u = i * m + j
                w = k * n + i
                eqs.append((u, w))
            V_indices.append((j, k, v, eqs))
            
    # For updating W:
    # For each w (k, i), we have equations for j=0..m-1
    # D_w[j, r] = U[i*m+j, r] * V[j*p+k, r]
    W_indices = []
    for k in range(p):
        for i in range(n):
            w = k * n + i
            eqs = []
            for j in range(m):
                u = i * m + j
                v = j * p + k
                eqs.append((u, v))
            W_indices.append((k, i, w, eqs))

    # Precompute the "active" indices for vectorized error calculation
    u_idxs = []
    v_idxs = []
    w_idxs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idxs.append(i * m + j)
                v_idxs.append(j * p + k)
                w_idxs.append(k * n + i)
    u_idxs = np.array(u_idxs)
    v_idxs = np.array(v_idxs)
    w_idxs = np.array(w_idxs)

    def compute_residuals(U, V, W):
        prod = U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :]
        sums = np.sum(prod, axis=1)
        return 1.0 - sums

    def verify_solution(U, V, W):
        # Check if U, V, W are half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9): return False
        
        # Check exact decomposition
        res = compute_residuals(U, V, W)
        if not np.allclose(res, 0.0, atol=1e-9):
            return False
        return True

    def als_decompose(R, max_iters=200, tol=1e-12):
        # Random initialization
        U = np.random.randint(-2, 3, (n * m, R)) * 0.5
        V = np.random.randint(-2, 3, (m * p, R)) * 0.5
        W = np.random.randint(-2, 3, (p * n, R)) * 0.5
        
        prev_err = np.inf
        for it in range(max_iters):
            # Update U
            for i, j, u, eqs in U_indices:
                B = np.zeros((len(eqs), R))
                for idx, (v, w) in enumerate(eqs):
                    B[idx, :] = V[v, :] * W[w, :]
                U_u, _, _, _ = np.linalg.lstsq(B, np.ones(len(eqs)), rcond=None)
                U[u, :] = U_u
                
            # Update V
            for j, k, v, eqs in V_indices:
                C = np.zeros((len(eqs), R))
                for idx, (u, w) in enumerate(eqs):
                    C[idx, :] = U[u, :] * W[w, :]
                V_v, _, _, _ = np.linalg.lstsq(C, np.ones(len(eqs)), rcond=None)
                V[v, :] = V_v
                
            # Update W
            for k, i, w, eqs in W_indices:
                D = np.zeros((len(eqs), R))
                for idx, (u, v) in enumerate(eqs):
                    D[idx, :] = U[u, :] * V[v, :]
                W_w, _, _, _ = np.linalg.lstsq(D, np.ones(len(eqs)), rcond=None)
                W[w, :] = W_w
            
            # Calculate error
            res = compute_residuals(U, V, W)
            err = np.sum(res ** 2)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        # Round to half-integers
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
            
        # Try a polish step if error is small
        res = compute_residuals(U, V, W)
        if np.sum(res**2) < 1e-4:
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residuals_func(x):
                U_flat = x[:n*m*R].reshape(n*m, R)
                V_flat = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_flat = x[n*m*R + m*p*R :].reshape(p*n, R)
                return compute_residuals(U_flat, V_flat, W_flat)
            
            try:
                res_polish = so.least_squares(residuals_func, x0, method='lm', max_nfev=1000)
                x_opt = res_polish.x
                U_opt = x_opt[:n*m*R].reshape(n*m, R)
                V_opt = x_opt[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_opt = x_opt[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Round and verify
                U_opt_r = np.round(U_opt * 2) / 2
                V_opt_r = np.round(V_opt * 2) / 2
                W_opt_r = np.round(W_opt * 2) / 2
                
                if verify_solution(U_opt_r, V_opt_r, W_opt_r):
                    return U_opt_r, V_opt_r, W_opt_r
            except Exception:
                pass

        return None

    start_time = time.time()
    deadline = 140 # Safety margin

    # Try rank 14, 15, 16 first
    for R in [14, 15, 16, 17, 18]:
        if time.time() > deadline:
            break
            
        # Try multiple restarts
        max_restarts = 200 if R <= 15 else 100
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            res = als_decompose(R)
            if res is not None:
                U, V, W = res
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
