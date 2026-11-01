import numpy as np
import time
import scipy.optimize

def construct():
    n, m, p = 2, 2, 7
    # Seed solution: Rank 25
    U0 = np.zeros((n*m, 25), dtype=float)
    V0 = np.zeros((m*p, 25), dtype=float)
    W0 = np.zeros((p*n, 25), dtype=float)
    
    # Reconstruct the rank 25 solution from the current code
    def add_strassen_block(U, V, W, col_start, k_start):
        # M1
        U[0, col_start+0] += 1; U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1; V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1; W[2*(k_start+1) + 1, col_start+0] += 1
        # M2
        U[2, col_start+1] += 1; U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1; W[2*(k_start+1) + 1, col_start+1] -= 1
        # M3
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1; V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1; W[2*(k_start+1) + 1, col_start+2] += 1
        # M4
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1; V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1; W[2*k_start + 1, col_start+3] += 1
        # M5
        U[0, col_start+4] += 1; U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1; W[2*(k_start+1) + 0, col_start+4] += 1
        # M6
        U[2, col_start+5] += 1; U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1; V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1
        # M7
        U[1, col_start+6] += 1; U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1; V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1

    add_strassen_block(U0, V0, W0, 0, 0)
    add_strassen_block(U0, V0, W0, 7, 2)
    add_strassen_block(U0, V0, W0, 14, 4)
    
    col_start = 21
    k = 6
    U0[0, col_start+0] += 1; V0[7*0 + k, col_start+0] += 1; W0[2*k + 0, col_start+0] += 1
    U0[1, col_start+1] += 1; V0[7*1 + k, col_start+1] += 1; W0[2*k + 0, col_start+1] += 1
    U0[2, col_start+2] += 1; V0[7*0 + k, col_start+2] += 1; W0[2*k + 1, col_start+2] += 1
    U0[3, col_start+3] += 1; V0[7*1 + k, col_start+3] += 1; W0[2*k + 1, col_start+3] += 1

    # Define Error Function
    def error_func(params):
        # params is a flat array of U, V, W
        R = len(params) // (n*m + m*p + p*n)
        U = params[0:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        
        err = 0.0
        # Valid terms
        for i in range(n):
            for j in range(m):
                for k_idx in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k_idx
                    c_idx = k_idx * n + i
                    s = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                    err += (1.0 - s) ** 2
        # Invalid terms
        for a_idx in range(n*m):
            for b_idx in range(m*p):
                for c_idx in range(p*n):
                    # Check if this is a valid term
                    i = a_idx // m
                    j = a_idx % m
                    j2 = b_idx // p
                    k_idx = b_idx % p
                    i2 = c_idx % n
                    k_idx2 = c_idx // n
                    
                    if j == j2 and i == i2 and k_idx == k_idx2:
                        continue # Skip valid
                    s = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                    err += s ** 2
        return err

    # Start with Rank 25
    current_rank = 25
    current_U = U0.copy()
    current_V = V0.copy()
    current_W = W0.copy()
    
    best_rank = 25
    best_solution = (current_U, current_V, current_W)
    
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    # Try to reduce rank
    for target_rank in [24, 23, 22, 21]:
        if time.time() > deadline:
            break
            
        success = False
        # Try random removals
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Initialize with current solution, zero out some terms
            # We keep 'target_rank' terms. We pick which ones to keep.
            # Strategy: Keep the most "important" terms? Or random?
            # Let's try random subsets of indices.
            indices = np.random.choice(current_rank, size=target_rank, replace=False)
            sorted_indices = np.sort(indices)
            
            U_init = np.zeros((n*m, target_rank))
            V_init = np.zeros((m*p, target_rank))
            W_init = np.zeros((p*n, target_rank))
            
            for new_i, old_i in enumerate(sorted_indices):
                U_init[:, new_i] = current_U[:, old_i]
                V_init[:, new_i] = current_V[:, old_i]
                W_init[:, new_i] = current_W[:, old_i]
                
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimize
            try:
                res = scipy.optimize.minimize(error_func, params_init, method='Nelder-Mead', 
                                             options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < 1e-6:
                    # Verify
                    params_opt = res.x
                    U_opt = params_opt[0:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = params_opt[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = params_opt[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Check exactness
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k_idx in range(p):
                                a_idx = i * m + j
                                b_idx = j * p + k_idx
                                c_idx = k_idx * n + i
                                s = np.sum(U_opt[a_idx, :] * V_opt[b_idx, :] * W_opt[c_idx, :])
                                if abs(s - 1.0) > 1e-5:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if valid:
                        # Check invalids
                        for a_idx in range(n*m):
                            for b_idx in range(m*p):
                                for c_idx in range(p*n):
                                    i = a_idx // m
                                    j = a_idx % m
                                    j2 = b_idx // p
                                    k_idx = b_idx % p
                                    i2 = c_idx % n
                                    k_idx2 = c_idx // n
                                    if j == j2 and i == i2 and k_idx == k_idx2:
                                        continue
                                    s = np.sum(U_opt[a_idx, :] * V_opt[b_idx, :] * W_opt[c_idx, :])
                                    if abs(s) > 1e-5:
                                        valid = False
                                        break
                            if not valid: break
                            if not valid: break
                        if valid:
                            success = True
                            current_U = U_opt
                            current_V = V_opt
                            current_W = W_opt
                            current_rank = target_rank
                            best_rank = target_rank
                            best_solution = (current_U, current_V, current_W)
                            break
            except Exception:
                pass
        
        if not success:
            # If we failed to reduce to target_rank, we might still try lower ranks?
            # No, if we can't do 24, we can't do 23 from this state.
            # But maybe a different random start works?
            # We break the loop to save time if we fail significantly.
            # However, we can try to restart from the original Rank 25 for the next target?
            # Let's just break if we failed to reduce by 1.
            break

    # Return best found
    U, V, W = best_solution
    # Convert to int if close
    U_int = np.round(U).astype(int)
    V_int = np.round(V).astype(int)
    W_int = np.round(W).astype(int)
    
    # Final verification of integer solution
    valid = True
    for i in range(n):
        for j in range(m):
            for k_idx in range(p):
                a_idx = i * m + j
                b_idx = j * p + k_idx
                c_idx = k_idx * n + i
                s = np.sum(U_int[a_idx, :] * V_int[b_idx, :] * W_int[c_idx, :])
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if not valid:
        # Fallback to original Rank 25
        U, V, W = U0, V0, W0
        U_int = np.round(U).astype(int)
        V_int = np.round(V).astype(int)
        W_int = np.round(W).astype(int)
        
    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
