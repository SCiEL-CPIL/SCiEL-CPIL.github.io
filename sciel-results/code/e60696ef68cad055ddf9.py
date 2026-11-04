"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import random
    import time
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    schoolbook_rank = nm * p  # 42
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Build list of non-zero indices in T
    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_indices.append((i * m + j, j * p + k, k * n + i))
    
    # Precompute arrays for fast vectorized error
    T_i = np.array([t[0] for t in T_indices])
    T_j = np.array([t[1] for t in T_indices])
    T_k = np.array([t[2] for t in T_indices])
    T_count = len(T_indices)
    
    nonzero_set = set(T_indices)
    all_zero_indices = []
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                if (i, j, k) not in nonzero_set:
                    all_zero_indices.append((i, j, k))
    random.shuffle(all_zero_indices)
    T_zero_sample = all_zero_indices[:300]
    
    Z_i = np.array([t[0] for t in T_zero_sample])
    Z_j = np.array([t[1] for t in T_zero_sample])
    Z_k = np.array([t[2] for t in T_zero_sample])
    Z_count = len(T_zero_sample)
    
    def compute_error(U, V, W):
        """Vectorized L1 error computation."""
        r = U.shape[1]
        # For T entries: sum_c U[i,c]*V[j,c]*W[k,c] should equal 1
        vals_T = np.zeros(T_count)
        for c in range(r):
            vals_T += U[T_i, c] * V[T_j, c] * W[T_k, c]
        err_T = np.abs(vals_T - 1.0).sum()
        
        # For zero entries: sum_c U[i,c]*V[j,c]*W[k,c] should equal 0
        vals_Z = np.zeros(Z_count)
        for c in range(r):
            vals_Z += U[Z_i, c] * V[Z_j, c] * W[Z_k, c]
        err_Z = np.abs(vals_Z).sum()
        
        return err_T + err_Z
    
    def full_verify(U, V, W):
        """Full verification of the decomposition."""
        r = U.shape[1]
        result = np.zeros((nm, mp, pn), dtype=np.int64)
        for c in range(r):
            result += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
        for (i, j, k) in T_indices:
            if result[i, j, k] != 1:
                return False
        for (i, j, k) in all_zero_indices:
            if result[i, j, k] != 0:
                return False
        return True
    
    def schoolbook():
        U = np.zeros((nm, schoolbook_rank), dtype=np.int64)
        V = np.zeros((mp, schoolbook_rank), dtype=np.int64)
        W = np.zeros((pn, schoolbook_rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = schoolbook()
    
    # Strategy: Try random initializations with simulated annealing
    # Key insight: use vectorized error for speed, try multiple ranks
    
    for target_rank in range(34, schoolbook_rank):
        if time.time() > deadline - 15:
            break
        
        found = False
        num_attempts = 30 if target_rank >= 38 else 15
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small integers
            U = np.random.randint(-1, 2, (nm, target_rank), dtype=np.int64)
            V = np.random.randint(-1, 2, (mp, target_rank), dtype=np.int64)
            W = np.random.randint(-1, 2, (pn, target_rank), dtype=np.int64)
            
            current_error = compute_error(U, V, W)
            
            if current_error == 0 and full_verify(U, V, W):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break
            
            temp = 100.0
            max_steps = 3000
            for step in range(max_steps):
                if time.time() > deadline - 10:
                    break
                if current_error == 0:
                    break
                
                factor_idx = random.randint(0, 2)
                delta = random.choice([-2, -1, 1, 2])
                
                if factor_idx == 0:
                    row = random.randint(0, nm - 1)
                    col = random.randint(0, target_rank - 1)
                    old_val = U[row, col]
                    U[row, col] = old_val + delta
                    new_error = compute_error(U, V, W)
                    if new_error <= current_error or random.random() < np.exp(-(new_error - current_error) / max(temp, 0.1)):
                        current_error = new_error
                    else:
                        U[row, col] = old_val
                elif factor_idx == 1:
                    row = random.randint(0, mp - 1)
                    col = random.randint(0, target_rank - 1)
                    old_val = V[row, col]
                    V[row, col] = old_val + delta
                    new_error = compute_error(U, V, W)
                    if new_error <= current_error or random.random() < np.exp(-(new_error - current_error) / max(temp, 0.1)):
                        current_error = new_error
                    else:
                        V[row, col] = old_val
                else:
                    row = random.randint(0, pn - 1)
                    col = random.randint(0, target_rank - 1)
                    old_val = W[row, col]
                    W[row, col] = old_val + delta
                    new_error = compute_error(U, V, W)
                    if new_error <= current_error or random.random() < np.exp(-(new_error - current_error) / max(temp, 0.1)):
                        current_error = new_error
                    else:
                        W[row, col] = old_val
                
                temp *= 0.997
            
            if current_error == 0 and full_verify(U, V, W):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break
        
        if not found:
            # Try a few more attempts with different structure
            if time.time() > deadline - 10:
                break
            for attempt2 in range(5):
                if time.time() > deadline - 10:
                    break
                U = np.random.randint(-1, 2, (nm, target_rank), dtype=np.int64)
                V = np.random.randint(-1, 2, (mp, target_rank), dtype=np.int64)
                W = np.random.randint(-1, 2, (pn, target_rank), dtype=np.int64)
                current_error = compute_error(U, V, W)
                
                temp = 200.0
                for step in range(5000):
                    if time.time() > deadline - 10:
                        break
                    if current_error == 0:
                        break
                    
                    factor_idx = random.randint(0, 2)
                    delta = random.choice([-2, -1, 1, 2])
                    
                    if factor_idx == 0:
                        row = random.randint(0, nm - 1)
                        col = random.randint(0, target_rank - 1)
                        old_val = U[row, col]
                        U[row, col] = old_val + delta
                        new_error = compute_error(U, V, W)
                        if new_error <= current_error or random.random() < np.exp(-(new_error - current_error) / max(temp, 0.1)):
                            current_error = new_error
                        else:
                            U[row, col] = old_val
                    elif factor_idx == 1:
                        row = random.randint(0, mp - 1)
                        col = random.randint(0, target_rank - 1)
                        old_val = V[row, col]
                        V[row, col] = old_val + delta
                        new_error = compute_error(U, V, W)
                        if new_error <= current_error or random.random() < np.exp(-(new_error - current_error) / max(temp, 0.1)):
                            current_error = new_error
                        else:
                            V[row, col] = old_val
                    else:
                        row = random.randint(0, pn - 1)
                        col = random.randint(0, target_rank - 1)
                        old_val = W[row, col]
                        W[row, col] = old_val + delta
                        new_error = compute_error(U, V, W)
                        if new_error <= current_error or random.random() < np.exp(-(new_error - current_error) / max(temp, 0.1)):
                            current_error = new_error
                        else:
                            W[row, col] = old_val
                    
                    temp *= 0.996
                
                if current_error == 0 and full_verify(U, V, W):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    found = True
                    break
            if found:
                break
        if not found:
            break
    
    U_list = [[int(x) for x in row] for row in best_U]
    V_list = [[int(x) for x in row] for row in best_V]
    W_list = [[int(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
