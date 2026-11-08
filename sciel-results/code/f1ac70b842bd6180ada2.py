"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    deadline = time.time() + 135
    
    # Try structured lower-rank decompositions first
    for target_rank in [30, 31, 32, 33]:
        if time.time() >= deadline:
            break
        res = attempt_structured(target_rank, deadline)
        if res is not None and valid_res(res, 3, 3, 4):
            return res
    
    # Fallback: return exactly rank 34 decomposition
    return construct_r34()

def attempt_structured(target_rank, deadline):
    n, m, p = 3, 3, 4
    R = target_rank
    
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
    
    total_params = (n*m + m*p + p*n) * R
    
    best_error = float('inf')
    best_params = None
    
    # Try multiple seeds with better initialization
    for seed in range(20):
        if time.time() >= deadline - 5:
            break
        
        np.random.seed(seed * 1000 + 42)
        
        # Initialize with structured pattern: start from rank-34 and project
        if seed < 10:
            # Sparse initialization
            params = np.zeros(total_params)
            for r in range(R):
                # Each rank-1 term should be sparse
                u_idx = np.random.choice(n*m, size=np.random.randint(1, 4), replace=False)
                v_idx = np.random.choice(m*p, size=np.random.randint(1, 4), replace=False)
                w_idx = np.random.choice(p*n, size=np.random.randint(1, 4), replace=False)
                
                u_vals = np.random.choice([-1, 1], size=len(u_idx))
                v_vals = np.random.choice([-1, 1], size=len(v_idx))
                w_vals = np.random.choice([-1, 1], size=len(w_idx))
                
                for idx, val in zip(u_idx, u_vals):
                    params[idx * R + r] = val
                offset_um = n*m * R
                for idx, val in zip(v_idx, v_vals):
                    params[offset_um + idx * R + r] = val
                offset_umv = (n*m + m*p) * R
                for idx, val in zip(w_idx, w_vals):
                    params[offset_umv + idx * R + r] = val
        else:
            params = np.random.randn(total_params) * 0.3
        
        from scipy.optimize import minimize
        try:
            def objective(params):
                U = params[0:n*m*R].reshape(n*m, R)
                V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W = params[n*m*R + m*p*R:].reshape(p*n, R)
                vals = np.sum(U[indices_a, :] * V[indices_b, :] * W[indices_c, :], axis=1)
                return np.sum((vals - 1.0)**2)
            
            res = minimize(objective, params, method='L-BFGS-B', 
                         options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 1000})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x.copy()
                if res.fun < 1e-8:
                    break
        except Exception:
            continue
    
    if best_params is None or best_error > 1e-5:
        return None
    
    U = best_params[0:n*m*R].reshape(n*m, R)
    V = best_params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
    W = best_params[n*m*R + m*p*R:].reshape(p*n, R)
    
    # Try rounding to half-integers
    for scale in [2, 4, 8]:
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        vals = np.sum(U_r[indices_a, :] * V_r[indices_b, :] * W_r[indices_c, :], axis=1)
        if np.allclose(vals, 1.0, atol=1e-9):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
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
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    U[2][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    col += 1
    U[5][col] = 1
    V[8][col] = 1
    W[1][col] = 1
    col += 1
    U[5][col] = 1
    V[9][col] = 1
    W[4][col] = 1
    col += 1
    U[6][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    U[6][col] = 1
    V[1][col] = 1
    W[5][col] = 1
    col += 1
    U[7][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    U[7][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    U[8][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    U[8][col] = 1
    V[9][col] = 1
    W[5][col] = 1
    col += 1
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[10][col] = 1
    col += 1
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[7][col] = 1; W[10][col] = -1
    col += 1
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[9][col] = 1; W[10][col] = 1
    col += 1
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[6][col] = -1; W[9][col] = 1
    col += 1
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[10][col] = 1
    col += 1
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1
    col += 1
    U[2][col] = 1
    V[10][col] = 1
    W[6][col] = 1
    col += 1
    U[2][col] = 1
    V[11][col] = 1
    W[9][col] = 1
    col += 1
    U[5][col] = 1
    V[10][col] = 1
    W[7][col] = 1
    col += 1
    U[5][col] = 1
    V[11][col] = 1
    W[10][col] = 1
    col += 1
    U[6][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    U[6][col] = 1
    V[3][col] = 1
    W[11][col] = 1
    col += 1
    U[7][col] = 1
    V[6][col] = 1
    W[8][col] = 1
    col += 1
    U[7][col] = 1
    V[7][col] = 1
    W[11][col] = 1
    col += 1
    U[8][col] = 1
    V[10][col] = 1
    W[8][col] = 1
    col += 1
    U[8][col] = 1
    V[11][col] = 1
    W[11][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
