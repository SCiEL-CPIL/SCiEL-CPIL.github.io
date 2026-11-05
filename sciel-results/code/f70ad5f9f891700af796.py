import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def _verify(U, V, W, R):
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            max_err = np.max(np.abs(T - 1.0))
            return max_err < 1e-8
        except:
            return False

    def _local_search(U, V, W, max_iters=500):
        R = U.shape[1]
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        err = T - 1.0
        max_err = np.max(np.abs(err))
        
        if max_err < 1e-8:
            return U, V, W, max_err
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            worst_idx = np.unravel_index(np.argmax(np.abs(err)), err.shape)
            wi, wj, wk = worst_idx
            werr = err[wi, wj, wk]
            
            r = np.random.randint(R)
            
            # Try adjusting U
            grad_u = V_t[wj, wk, r] * W_t[wk, wi, r]
            if grad_u != 0:
                old_u = U_t[wi, wj, r]
                direction = -np.sign(werr) * np.sign(grad_u)
                candidates = [old_u + direction * 0.5, old_u - direction * 0.5, 
                              old_u + direction * 1.0, old_u - direction * 1.0]
                
                best_candidate = old_u
                best_candidate_err = max_err + 1e-9
                
                for c in candidates:
                    c_snap = round(c * 2) / 2.0
                    if c_snap == old_u:
                        continue
                    U_t[wi, wj, r] = c_snap
                    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                    new_err = np.max(np.abs(T - 1.0))
                    if new_err < best_candidate_err:
                        best_candidate_err = new_err
                        best_candidate = c_snap
                    U_t[wi, wj, r] = old_u
                
                if best_candidate != old_u:
                    U_t[wi, wj, r] = best_candidate
                    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                    err = T - 1.0
                    max_err = np.max(np.abs(err))
                    if max_err < 1e-8:
                        return U, V, W, max_err
                    continue
            
            # Try adjusting V
            grad_v = U_t[wi, wj, r] * W_t[wk, wi, r]
            if grad_v != 0:
                old_v = V_t[wj, wk, r]
                direction = -np.sign(werr) * np.sign(grad_v)
                candidates = [old_v + direction * 0.5, old_v - direction * 0.5, 
                              old_v + direction * 1.0, old_v - direction * 1.0]
                
                best_candidate = old_v
                best_candidate_err = max_err + 1e-9
                
                for c in candidates:
                    c_snap = round(c * 2) / 2.0
                    if c_snap == old_v:
                        continue
                    V_t[wj, wk, r] = c_snap
                    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                    new_err = np.max(np.abs(T - 1.0))
                    if new_err < best_candidate_err:
                        best_candidate_err = new_err
                        best_candidate = c_snap
                    V_t[wj, wk, r] = old_v
                
                if best_candidate != old_v:
                    V_t[wj, wk, r] = best_candidate
                    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                    err = T - 1.0
                    max_err = np.max(np.abs(err))
                    if max_err < 1e-8:
                        return U, V, W, max_err
                    continue
            
            # Try adjusting W
            grad_w = U_t[wi, wj, r] * V_t[wj, wk, r]
            if grad_w != 0:
                old_w = W_t[wk, wi, r]
                direction = -np.sign(werr) * np.sign(grad_w)
                candidates = [old_w + direction * 0.5, old_w - direction * 0.5, 
                              old_w + direction * 1.0, old_w - direction * 1.0]
                
                best_candidate = old_w
                best_candidate_err = max_err + 1e-9
                
                for c in candidates:
                    c_snap = round(c * 2) / 2.0
                    if c_snap == old_w:
                        continue
                    W_t[wk, wi, r] = c_snap
                    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                    new_err = np.max(np.abs(T - 1.0))
                    if new_err < best_candidate_err:
                        best_candidate_err = new_err
                        best_candidate = c_snap
                    W_t[wk, wi, r] = old_w
                
                if best_candidate != old_w:
                    W_t[wk, wi, r] = best_candidate
                    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                    err = T - 1.0
                    max_err = np.max(np.abs(err))
                    if max_err < 1e-8:
                        return U, V, W, max_err
                    continue
            
            if np.random.random() < 0.1:
                for _ in range(3):
                    idx_u = np.random.randint(n*m)
                    r_rand = np.random.randint(R)
                    U[idx_u, r_rand] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                    err = T - 1.0
                    max_err = np.max(np.abs(err))
                    if max_err < 1e-8:
                        return U, V, W, max_err

        return U, V, W, max_err

    # Try ranks from 24 to 35
    # We prioritize lower ranks but spend limited time on each
    # Rank 24 is the target.
    for R in range(24, rank_school):
        if time.time() > deadline:
            break
        
        # Time allocation: 5 seconds per rank
        time_limit = time.time() + 5.0
        
        num_trials = 20
        for trial in range(num_trials):
            if time.time() > time_limit:
                break
            
            # Initialize with noise around schoolbook-like structure or random
            # Random half-integers
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n * m, R))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m * p, R))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p * n, R))
            
            U, V, W, err = _local_search(U, V, W, max_iters=200)
            
            if err < 1e-8:
                valid = _verify(U, V, W, R)
                if valid:
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    def snap(x):
                        return round(x * 2) / 2.0
                    U_list = [[snap(x) for x in row] for row in U_list]
                    V_list = [[snap(x) for x in row] for row in V_list]
                    W_list = [[snap(x) for x in row] for row in W_list]
                    return {"U": U_list, "V": V_list, "W": W_list}

    return best_result
