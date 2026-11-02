import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def _validate(U, V, W, n, m, p):
    """Exact check for tensor decomposition of matrix multiplication."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

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

def _search_rank17(n, m, p, max_time=110):
    start = time.time()
    target_rank = 17
    
    # Indices for the tensor T[i, j, k]
    idx_u = []
    idx_v = []
    idx_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u.append(i * m + j)
                idx_v.append(j * p + k)
                idx_w.append(k * n + i)
    idx_u = np.array(idx_u)
    idx_v = np.array(idx_v)
    idx_w = np.array(idx_w)
    
    def error_func(x):
        U_flat = x[:n*m*target_rank]
        V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = x[n*m*target_rank + m*p*target_rank :]
        
        U = U_flat.reshape(n*m, target_rank)
        V = V_flat.reshape(m*p, target_rank)
        W = W_flat.reshape(p*n, target_rank)
        
        # Compute sum U[idx_u] * V[idx_v] * W[idx_w]
        # This is efficient using numpy broadcasting if we construct the matrix properly, 
        # but for small sizes, a loop or vectorized sum is fine.
        # Let's vectorize:
        # We need sum over c of U[i,j,c] V[j,k,c] W[k,i,c]
        # U[idx_u, :] is shape (15, 17)
        # V[idx_v, :] is shape (15, 17)
        # W[idx_w, :] is shape (15, 17)
        
        U_sel = U[idx_u, :]
        V_sel = V[idx_v, :]
        W_sel = W[idx_w, :]
        
        # Element-wise product sum over last axis
        prod = U_sel * V_sel * W_sel
        sums = np.sum(prod, axis=1)
        
        # Target is 1.0 for all 15 entries
        diff = sums - 1.0
        return np.sum(diff * diff)

    def local_search_int(U, V, W, max_iters=1000):
        # U, V, W are numpy arrays of floats (currently rounded)
        # We will perform hill climbing on integer/half-integer grid
        best_err = error_func(np.concatenate([U.flatten(), V.flatten(), W.flatten()]))
        current_U = U.copy()
        current_V = V.copy()
        current_W = W.copy()
        
        for _ in range(max_iters):
            if time.time() - start > max_time:
                break
            
            # Try random perturbation
            r = random.random()
            if r < 0.33:
                row = random.randint(0, n*m - 1)
                col = random.randint(0, target_rank - 1)
                delta = random.choice([-0.5, 0.5, -1.0, 1.0])
                new_val = current_U[row][col] + delta
                # Keep within reasonable bounds
                if -10 < new_val < 10:
                    current_U[row][col] = new_val
            elif r < 0.66:
                row = random.randint(0, m*p - 1)
                col = random.randint(0, target_rank - 1)
                delta = random.choice([-0.5, 0.5, -1.0, 1.0])
                new_val = current_V[row][col] + delta
                if -10 < new_val < 10:
                    current_V[row][col] = new_val
            else:
                row = random.randint(0, p*n - 1)
                col = random.randint(0, target_rank - 1)
                delta = random.choice([-0.5, 0.5, -1.0, 1.0])
                new_val = current_W[row][col] + delta
                if -10 < new_val < 10:
                    current_W[row][col] = new_val
            
            new_err = error_func(np.concatenate([current_U.flatten(), current_V.flatten(), current_W.flatten()]))
            
            if new_err < best_err - 1e-9:
                best_err = new_err
                if best_err < 1e-6:
                    # Check exact validity
                    U_snap = np.round(current_U * 2.0) / 2.0
                    V_snap = np.round(current_V * 2.0) / 2.0
                    W_snap = np.round(current_W * 2.0) / 2.0
                    if _validate(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
            else:
                # Revert
                if r < 0.33:
                    current_U[row][col] -= delta
                elif r < 0.66:
                    current_V[row][col] -= delta
                else:
                    current_W[row][col] -= delta
            
            if time.time() - start > max_time:
                break

        return None

    for seed in range(100):
        if time.time() - start > max_time:
            break
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with small integers
        U_init = np.random.choice([-1, 0, 1], size=(n*m, target_rank)).astype(float)
        V_init = np.random.choice([-1, 0, 1], size=(m*p, target_rank)).astype(float)
        W_init = np.random.choice([-1, 0, 1], size=(p*n, target_rank)).astype(float)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Continuous optimization to get close to solution
        try:
            res = minimize(error_func, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        except Exception:
            continue
            
        x_opt = res.x
        U_opt = x_opt[:n*m*target_rank].reshape(n*m, target_rank)
        V_opt = x_opt[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_opt = x_opt[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        
        # Round to nearest 0.5
        U_snap = np.round(U_opt * 2.0) / 2.0
        V_snap = np.round(V_opt * 2.0) / 2.0
        W_snap = np.round(W_opt * 2.0) / 2.0
        
        # Check if direct rounding works
        if _validate(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
            return {
                "U": U_snap.tolist(),
                "V": V_snap.tolist(),
                "W": W_snap.tolist()
            }
            
        # If not, try local search from this point
        result = local_search_int(U_snap, V_snap, W_snap)
        if result:
            return result

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try search
    result = _search_rank17(n, m, p, max_time=110)
    if result is not None:
        if _validate(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fallback
    return _schoolbook(n, m, p)
