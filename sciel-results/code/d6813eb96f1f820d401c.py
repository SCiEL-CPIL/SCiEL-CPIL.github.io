import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    T = np.ones((n, m, p))
    
    # Schoolbook fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    start_time = time.time()
    deadline = 135
    
    def compute_error(U, V, W, R):
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.linalg.norm(result - T)
    
    def als_step(U, V, W, R):
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, R))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    U[i * m + j, :] = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)[0]
                except:
                    pass
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, R))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :] = np.linalg.lstsq(M.T, T[:, j, k], rcond=None)[0]
                except:
                    pass
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, R))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :] = np.linalg.lstsq(M.T, T[i, :, k], rcond=None)[0]
                except:
                    pass
    
    # Try rank 14 with multiple strategies
    target_rank = 14
    found = False
    
    # Strategy 1: Random initialization
    for restart in range(100):
        if time.time() - start_time > deadline or found:
            break
        rng = np.random.RandomState(restart * 7919)
        U = rng.randn(n * m, target_rank)
        V = rng.randn(m * p, target_rank)
        W = rng.randn(p * n, target_rank)
        
        prev_err = np.inf
        for it in range(500):
            if time.time() - start_time > deadline:
                break
            als_step(U, V, W, target_rank)
            err = compute_error(U, V, W, target_rank)
            if err < 1e-10:
                found = True
                break
            if abs(err - prev_err) < 1e-12 and it > 10:
                U += 0.3 * rng.randn(n * m, target_rank)
                V += 0.3 * rng.randn(m * p, target_rank)
                W += 0.3 * rng.randn(p * n, target_rank)
            prev_err = err
    
    # Strategy 2: Start from schoolbook with 4 columns zeroed
    if not found and time.time() - start_time < deadline:
        for restart in range(50):
            if time.time() - start_time > deadline:
                break
            rng = np.random.RandomState(10000 + restart)
            
            # Remove 4 random columns from schoolbook
            cols_to_keep = list(range(R_school))
            rng.shuffle(cols_to_keep)
            cols_to_keep = cols_to_keep[:target_rank]
            cols_to_keep.sort()
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            for new_col, old_col in enumerate(cols_to_keep):
                U[:, new_col] = U_school[:, old_col]
                V[:, new_col] = V_school[:, old_col]
                W[:, new_col] = W_school[:, old_col]
            
            # Add small random perturbation
            U += 0.1 * rng.randn(n * m, target_rank)
            V += 0.1 * rng.randn(m * p, target_rank)
            W += 0.1 * rng.randn(p * n, target_rank)
            
            prev_err = np.inf
            for it in range(500):
                if time.time() - start_time > deadline:
                    break
                als_step(U, V, W, target_rank)
                err = compute_error(U, V, W, target_rank)
                if err < 1e-10:
                    found = True
                    break
                if abs(err - prev_err) < 1e-12 and it > 10:
                    U += 0.2 * rng.randn(n * m, target_rank)
                    V += 0.2 * rng.randn(m * p, target_rank)
                    W += 0.2 * rng.randn(p * n, target_rank)
                prev_err = err
    
    if found:
        for scale in [1, 2, 4, 8, 16, 32]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if compute_error(U_r, V_r, W_r, target_rank) < 1e-8:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                break
        else:
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
    
    # If rank 14 failed, try 15
    if best_rank == R_school and time.time() - start_time < deadline:
        target_rank = 15
        for restart in range(50):
            if time.time() - start_time > deadline:
                break
            rng = np.random.RandomState(20000 + restart)
            U = rng.randn(n * m, target_rank)
            V = rng.randn(m * p, target_rank)
            W = rng.randn(p * n, target_rank)
            
            prev_err = np.inf
            for it in range(500):
                if time.time() - start_time > deadline:
                    break
                als_step(U, V, W, target_rank)
                err = compute_error(U, V, W, target_rank)
                if err < 1e-10:
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    break
                if abs(err - prev_err) < 1e-12 and it > 10:
                    U += 0.3 * rng.randn(n * m, target_rank)
                    V += 0.3 * rng.randn(m * p, target_rank)
                    W += 0.3 * rng.randn(p * n, target_rank)
                prev_err = err
            if best_rank == target_rank:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
