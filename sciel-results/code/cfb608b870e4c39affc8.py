import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_schoolbook
    best_error = float('inf')
    
    # Initialize with schoolbook
    U0 = np.zeros((n*m, rank_schoolbook))
    V0 = np.zeros((m*p, rank_schoolbook))
    W0 = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
                
    def compute_error(U, V, W, r):
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        target = np.ones((n, m, p))
        error = np.sum((T - target) ** 2)
        return error

    err0 = compute_error(U0, V0, W0, rank_schoolbook)
    if err0 < 1e-9:
        best_error = err0
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank_schoolbook

    # Try rank 17
    target_rank = 17
    if target_rank < best_rank:
        for restart in range(20):
            if time.time() > deadline - 10:
                break
                
            U = np.random.uniform(-2, 2, size=(n*m, target_rank))
            V = np.random.uniform(-2, 2, size=(m*p, target_rank))
            W = np.random.uniform(-2, 2, size=(p*n, target_rank))
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat, V_flat, W_flat = x[:n*m*target_rank], x[n*m*target_rank:n*m*target_rank + m*p*target_rank], x[n*m*target_rank + m*p*target_rank:]
                U_curr = U_flat.reshape(n*m, target_rank)
                V_curr = V_flat.reshape(m*p, target_rank)
                W_curr = W_flat.reshape(p*n, target_rank)
                return compute_error(U_curr, V_curr, W_curr, target_rank)
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            except Exception:
                continue
                
            U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            err_snap = compute_error(U_snap, V_snap, W_snap, target_rank)
            
            if err_snap < best_error:
                best_error = err_snap
                best_U, best_V, best_W = U_snap, V_snap, W_snap
                best_rank = target_rank
                
                if best_error < 1e-9:
                    break

    # If rank 17 not found, try rank 16
    if best_rank > 16 and time.time() < deadline - 20:
        target_rank = 16
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            U = np.random.uniform(-2, 2, size=(n*m, target_rank))
            V = np.random.uniform(-2, 2, size=(m*p, target_rank))
            W = np.random.uniform(-2, 2, size=(p*n, target_rank))
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat, V_flat, W_flat = x[:n*m*target_rank], x[n*m*target_rank:n*m*target_rank + m*p*target_rank], x[n*m*target_rank + m*p*target_rank:]
                U_curr = U_flat.reshape(n*m, target_rank)
                V_curr = V_flat.reshape(m*p, target_rank)
                W_curr = W_flat.reshape(p*n, target_rank)
                return compute_error(U_curr, V_curr, W_curr, target_rank)
                
            try:
                res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            except Exception:
                continue
                
            U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            err_snap = compute_error(U_snap, V_snap, W_snap, target_rank)
            
            if err_snap < best_error:
                best_error = err_snap
                best_U, best_V, best_W = U_snap, V_snap, W_snap
                best_rank = target_rank
                if best_error < 1e-9:
                    break

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    def clean_list(lst):
        cleaned = []
        for row in lst:
            new_row = []
            for val in row:
                nearest = round(val * 2) / 2
                new_row.append(nearest)
            cleaned.append(new_row)
        return cleaned

    U_list = clean_list(U_list)
    V_list = clean_list(V_list)
    W_list = clean_list(W_list)

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
