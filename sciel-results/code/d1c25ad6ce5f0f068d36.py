import numpy as np
import time
import sys
from scipy.linalg import null_space

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
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Check if all T[i, j, k] == 1
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        
        try:
            # U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
            # einsum indices: i j r, j k r, k i r -> i j k
            T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            max_err = np.max(np.abs(T - 1.0))
            return max_err < 1e-8
        except:
            return False

    def _local_search(U, V, W, max_iters=2000):
        R = U.shape[1]
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        
        # Compute initial error
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        err = T - 1.0
        max_err = np.max(np.abs(err))
        
        if max_err < 1e-8:
            return U, V, W, max_err
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find worst index
            worst_idx = np.unravel_index(np.argmax(np.abs(err)), err.shape)
            wi, wj, wk = worst_idx
            werr = err[wi, wj, wk]
            
            # Try to adjust one entry in U, V, or W
            # We will try a random column r and adjust the corresponding entry
            r = np.random.randint(R)
            
            # 1. Try adjusting U[wi, wj, r]
            # d/dU[wi, wj, r] T[wi, wj, wk] = V[wj, wk, r] * W[wk, wi, r]
            grad_u = V_t[wj, wk, r] * W_t[wk, wi, r]
            if grad_u != 0:
                old_u = U_t[wi, wj, r]
                # Try small half-integer steps
                # If werr > 0, we want to decrease T. If grad_u > 0, decrease U.
                # If werr < 0, we want to increase T. If grad_u > 0, increase U.
                # Direction: -sign(werr) * sign(grad_u)
                # Step size: 0.5
                step = 0.5
                direction = -np.sign(werr) * np.sign(grad_u)
                
                # Try a few candidates
                candidates = [old_u + direction * step, old_u - direction * step, 
                              old_u + direction * 1.0, old_u - direction * 1.0]
                
                best_candidate = old_u
                best_candidate_err = max_err + 1e-9 # Must improve global max err
                
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
                    U_t[wi, wj, r] = old_u # Restore
                
                if best_candidate != old_u:
                    U_t[wi, wj, r] = best_candidate
                    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                    err = T - 1.0
                    max_err = np.max(np.abs(err))
                    if max_err < 1e-8:
                        return U, V, W, max_err
                    continue # Restart loop with new error
            
            # 2. Try adjusting V[wj, wk, r]
            # d/dV[wj, wk, r] T[wi, wj, wk] = U[wi, wj, r] * W[wk, wi, r]
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
                    V_t[wj, wk, r] = old_v # Restore
                
                if best_candidate != old_v:
                    V_t[wj, wk, r] = best_candidate
                    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                    err = T - 1.0
                    max_err = np.max(np.abs(err))
                    if max_err < 1e-8:
                        return U, V, W, max_err
                    continue # Restart loop with new error
            
            # 3. Try adjusting W[wk, wi, r]
            # d/dW[wk, wi, r] T[wi, wj, wk] = U[wi, wj, r] * V[wj, wk, r]
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
                    W_t[wk, wi, r] = old_w # Restore
                
                if best_candidate != old_w:
                    W_t[wk, wi, r] = best_candidate
                    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                    err = T - 1.0
                    max_err = np.max(np.abs(err))
                    if max_err < 1e-8:
                        return U, V, W, max_err
                    continue # Restart loop with new error
            
            # If no move improved, add random perturbation to escape local minima
            if np.random.random() < 0.1:
                # Random restart of a few entries
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

    # Try to find lower rank
    # Start from 20 as it is a known lower bound/achievable rank for 3x3x4
    # We iterate R from 20 upwards until we find a solution or time runs out
    for R in range(20, rank_school):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        # We limit trials per rank to ensure we can try lower ranks if they are easier?
        # No, we want to find the MINIMAL rank. So we must succeed at R=20 before trying R=21.
        # If we fail at R=20 after X trials, we move to R=21? 
        # The problem asks for rank-minimal. So we should exhaust R=20 before R=21.
        # But 150s is tight. 
        # Strategy: Spend a fraction of time on each rank? 
        # Or just try R=20 for a while, then R=21, etc.
        # Since 20 is the suspected minimum, we should spend most time there.
        
        num_trials = 100
        # Reduce trials if time is running low
        time_per_rank = (deadline - time.time()) / (rank_school - R)
        if time_per_rank < 1.0:
            break
            
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            # Random init with small integers/half-integers
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n * m, R))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m * p, R))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p * n, R))
            
            # Run local search
            U, V, W, err = _local_search(U, V, W, max_iters=1000)
            
            if err < 1e-8:
                # Verify and return
                valid = _verify(U, V, W, R)
                if valid:
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    # Snap to exact half-integers to avoid floating point representation issues in JSON
                    def snap(x):
                        return round(x * 2) / 2.0
                    U_list = [[snap(x) for x in row] for row in U_list]
                    V_list = [[snap(x) for x in row] for row in V_list]
                    W_list = [[snap(x) for x in row] for row in W_list]
                    return {"U": U_list, "V": V_list, "W": W_list}

    return best_result
