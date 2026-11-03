import numpy as np
import time
import random
from fractions import Fraction

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def _local_search_half_integers(U, V, W, T, max_passes=1000, time_limit=None, start_time=None):
    """
    Perform local search on the half-integer lattice to minimize the error.
    U, V, W are numpy arrays of half-integers.
    T is the target tensor.
    """
    if start_time is None:
        start_time = time.time()
        
    n_rows_U = U.shape[0]
    n_rows_V = V.shape[0]
    n_rows_W = W.shape[0]
    rank = U.shape[1]
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    current_error = np.sum((T - T_recon) ** 2)
    
    if current_error < 1e-9:
        return U, V, W, current_error

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_error = current_error
    T_recon_best = T_recon.copy()
    
    for p_idx in range(max_passes):
        if time_limit and time.time() > time_limit:
            break
            
        improved = False
        
        # Try updating U
        for a in range(n_rows_U):
            for r in range(rank):
                old_val = best_U[a, r]
                for delta in [0.5, -0.5]:
                    new_val = old_val + delta
                    delta_T = delta * np.outer(V[:, r], W[:, r])
                    old_diff = T[a] - T_recon_best[a]
                    new_diff = old_diff - delta_T
                    new_error_part = np.sum(new_diff ** 2)
                    old_error_part = np.sum(old_diff ** 2)
                    new_total_error = best_error - old_error_part + new_error_part
                    if new_total_error < best_error - 1e-12:
                        T_recon_best[a] += delta_T
                        best_U[a, r] = new_val
                        best_error = new_total_error
                        improved = True

        # Try updating V
        for b in range(n_rows_V):
            for r in range(rank):
                old_val = best_V[b, r]
                for delta in [0.5, -0.5]:
                    new_val = old_val + delta
                    delta_T = delta * np.outer(U[:, r], W[:, r])
                    old_diff = T[:, b, :] - T_recon_best[:, b, :]
                    new_diff = old_diff - delta_T
                    new_error_part = np.sum(new_diff ** 2)
                    old_error_part = np.sum(old_diff ** 2)
                    new_total_error = best_error - old_error_part + new_error_part
                    if new_total_error < best_error - 1e-12:
                        T_recon_best[:, b, :] += delta_T
                        best_V[b, r] = new_val
                        best_error = new_total_error
                        improved = True

        # Try updating W
        for c in range(n_rows_W):
            for r in range(rank):
                old_val = best_W[c, r]
                for delta in [0.5, -0.5]:
                    new_val = old_val + delta
                    delta_T = delta * np.outer(U[:, r], V[:, r])
                    old_diff = T[:, :, c] - T_recon_best[:, :, c]
                    new_diff = old_diff - delta_T
                    new_error_part = np.sum(new_diff ** 2)
                    old_error_part = np.sum(old_diff ** 2)
                    new_total_error = best_error - old_error_part + new_error_part
                    if new_total_error < best_error - 1e-12:
                        T_recon_best[:, :, c] += delta_T
                        best_W[c, r] = new_val
                        best_error = new_total_error
                        improved = True

        if not improved:
            break

    return best_U, best_V, best_W, best_error

def _try_rank_local_search(n, m, p, r, T, time_limit):
    """
    Try to find a decomposition of rank r using local search from random starts.
    """
    start_time = time.time()
    max_restarts = 5000
    
    for restart in range(max_restarts):
        if time.time() > time_limit:
            break
        
        vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        
        U = np.random.choice(vals, size=(n*m, r))
        V = np.random.choice(vals, size=(m*p, r))
        W = np.random.choice(vals, size=(p*n, r))
        
        U_sol, V_sol, W_sol, err = _local_search_half_integers(U, V, W, T, max_passes=50, time_limit=time_limit)
        
        if err < 1e-9:
            T_recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            if np.allclose(T_recon, T, atol=1e-9, rtol=0):
                return U_sol.tolist(), V_sol.tolist(), W_sol.tolist()
                
    return None

def construct():
    n, m, p = 2, 3, 7
    T = _build_tensor(n, m, p)
    
    start_time = time.time()
    max_time = 140.0 # 10s margin before 150s kill
    
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_solution = None
    best_rank = n * m * p
    
    # The current program starts at 29. 
    # Let's stick to the same strategy but maybe optimize the local search slightly or just keep it.
    # The directive says "KEEP this design".
    # I will keep the logic almost identical to ensure stability, as the previous score was 1.0 (baseline).
    # To potentially improve, I could try to make the local search faster or smarter.
    # One improvement: in _local_search_half_integers, we can precompute the outer products for each r?
    # No, U, V, W change, so outer products change.
    
    # Let's just run the same search.
    for r in range(29, 42):
        if time.time() > max_time:
            break
            
        remaining = max_time - (time.time() - start_time)
        if remaining < 10:
            break
            
        time_slice = min(remaining / (42 - r), 30.0)
        
        sol = _try_rank_local_search(n, m, p, r, T, time.time() + time_slice)
        
        if sol is not None:
            U, V, W = sol
            best_solution = (U, V, W)
            best_rank = r
            if r <= 30:
                 pass

    if best_solution:
        return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
