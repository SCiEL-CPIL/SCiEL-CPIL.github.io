import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    # Rank 50 decomposition (Strassen-based)
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Part 1: A_L (2x2) * B_T (2x9)
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            # Strassen
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        # Last col of B_T
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        # Part 2: A_R (2x1) * B_B (1x9)
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}

    def verify_fast(U, V, W):
        """Fast verification using numpy"""
        r = len(U[0])
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if val != 1:
                        return False
        return True

    def verify_partial(U, V, W, i, j):
        """Check if entry (i,j) is satisfied"""
        r = len(U[0])
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        for k in range(p):
            val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
            if val != 1:
                return False
        return True

    def try_rank_local_search(r, base_U, base_V, base_W, max_trials=200):
        """Local search starting from a perturbed version of the known solution"""
        if time.time() > deadline - 15:
            return None
        
        base_U = np.array(base_U, dtype=np.int64)
        base_V = np.array(base_V, dtype=np.int64)
        base_W = np.array(base_W, dtype=np.int64)
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            # Start from the rank-50 solution and try to remove columns
            # by merging or zeroing out
            U = base_U[:, :r].copy()
            V = base_V[:, :r].copy()
            W = base_W[:, :r].copy()
            
            # Try to find columns that can be removed or merged
            # Strategy: identify columns with similar patterns and try to merge
            success = True
            for attempt in range(20):
                if time.time() > deadline - 10:
                    return None
                
                # Find pairs of columns that might be mergeable
                # Two columns c1, c2 can be merged if there exists a scalar s
                # such that V[:,c1] + s*V[:,c2] and W[:,c1] + s*W[:,c2]
                # still allow U to be solved exactly
                c1, c2 = random.sample(range(r), 2)
                
                # Try to eliminate column c2 by expressing it in terms of c1
                # We need: for all (i,j,k): U[i,j,c1]*V[j,k,c1]*W[k,i,c1] + 
                #   U[i,j,c2]*V[j,k,c2]*W[k,i,c2] -> replace with modified c1
                # This is complex. Instead, try simple mutations:
                
                # Mutation 1: Add a small perturbation to V and W, re-solve U
                V_new = V.copy()
                W_new = W.copy()
                
                # Perturb 1-3 entries in V
                n_perturb = random.randint(1, 3)
                for _ in range(n_perturb):
                    row = random.randint(0, m_rows-1)
                    col = random.randint(0, r-1)
                    V_new[row, col] += random.choice([-1, 1])
                    if V_new[row, col] == 0:
                        V_new[row, col] = random.choice([-1, 1])
                
                # Perturb 1-3 entries in W
                for _ in range(n_perturb):
                    row = random.randint(0, p_rows-1)
                    col = random.randint(0, r-1)
                    W_new[row, col] += random.choice([-1, 1])
                    if W_new[row, col] == 0:
                        W_new[row, col] = random.choice([-1, 1])
                
                # Re-solve for U
                U_new = np.zeros((n_rows, r), dtype=np.int64)
                valid = True
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, r), dtype=np.float64)
                        for k in range(p):
                            M[k, :] = V_new[j*p+k, :] * W_new[k*n+i, :]
                        
                        try:
                            u, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        except:
                            valid = False
                            break
                        
                        u_int = np.round(u).astype(np.int64)
                        
                        if not np.allclose(M @ u_int, np.ones(p), atol=1e-10):
                            valid = False
                            break
                            
                        U_new[i*m+j, :] = u_int
                    
                    if not valid:
                        break
                
                if valid and verify_fast(U_new, V_new, W_new):
                    return {
                        "U": U_new.tolist(),
                        "V": V_new.tolist(),
                        "W": W_new.tolist()
                    }
            
            # Also try: start from scratch with structured approach
            if trial % 5 == 0:
                result = try_rank_structured(r)
                if result:
                    return result
                    
        return None

    def try_rank_structured(r, max_trials=50):
        """Structured approach based on block decomposition"""
        if time.time() > deadline - 15:
            return None
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            V = np.zeros((m_rows, r), dtype=np.int64)
            W = np.zeros((p_rows, r), dtype=np.int64)
            
            # Assign columns to blocks based on the known structure
            # 28 columns for 4 Strassen blocks (7 each), 4 for last col, 18 for A_R*B_B
            # Total = 50. For lower rank, we need to share columns across blocks.
            
            # Strategy: use shared columns for the A_R*B_B part
            # The A_R*B_B part needs 18 columns (2 per k). 
            # Can we share with Strassen columns?
            
            # Let's try: assign first 28 columns to Strassen (4*7), 
            # then try to fit A_R*B_B into remaining r-28 columns
            # But that only works if r >= 28+18 = 46
            
            # Better: interleave and share
            # For each k in 0..8, we need columns for:
            # - Strassen (if k < 8): 7 columns per pair (k0, k1)
            # - A_R*B_B: 2 columns (one for i=0, one for i=1)
            # - Last col (k=8): 4 columns
            
            # Let's try a different approach: random assignment with constraints
            for c in range(r):
                # Determine which block this column serves
                block = c % 3  # 0: Strassen, 1: last col, 2: A_R*B_B
                
                if block == 0:
                    # Strassen: 3-4 V entries in rows 0-17, 2-3 W entries
                    v_count = random.choice([3, 4])
                    v_indices = list(np.random.choice(18, size=min(v_count, 18), replace=False))
                    for idx in v_indices:
                        V[idx, c] = random.choice([-1, 1])
                    
                    w_count = random.choice([2, 3])
                    k_vals = np.random.choice(9, size=min(w_count, 9), replace=False)
                    for k in k_vals:
                        W[k * n + random.choice([0, 1]), c] = random.choice([-1, 1])
                        
                elif block == 1:
                    # Last col: 2 V entries in rows 0-17 (k=8), 2 W entries
                    v_indices = list(np.random.choice([0*9+8, 1*9+8], size=2, replace=False))
                    for idx in v_indices:
                        V[idx, c] = random.choice([-1, 1])
                    
                    W[8*n+0, c] = random.choice([-1, 1])
                    W[8*n+1, c] = random.choice([-1, 1])
                        
                else:
                    # A_R*B_B: 1 V entry in rows 18-26, 1 W entry
                    v_idx = random.randint(18, 26)
                    V[v_idx, c] = random.choice([-1, 1])
                    
                    k = v_idx - 18
                    W[k*n + random.choice([0, 1]), c] = random.choice([-1, 1])
            
            # Solve for U
            U = np.zeros((n_rows, r), dtype=np.int64)
            valid = True
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r), dtype=np.float64)
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    
                    try:
                        u, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    except:
                        valid = False
                        break
                    
                    u_int = np.round(u).astype(np.int64)
                    
                    if not np.allclose(M @ u_int, np.ones(p), atol=1e-10):
                        valid = False
                        break
                        
                    U[i*m+j, :] = u_int
                
            if not valid:
                continue
            
            if verify_fast(U, V, W):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
        return None

    # Get the base solution
    base = current_solution()
    base_U = np.array(base["U"], dtype=np.int64)
    base_V = np.array(base["V"], dtype=np.int64)
    base_W = np.array(base["W"], dtype=np.int64)
    
    best_result = None
    
    # Try local search first (more likely to succeed)
    for r in range(49, 45, -1):
        if time.time() > deadline - 30:
            break
        trials = max(30, 100 - (49 - r) * 15)
        res = try_rank_local_search(r, base["U"], base["V"], base["W"], max_trials=trials)
        if res:
            best_result = res
            break
        
    # If local search didn't work, try structured random
    if not best_result:
        for r in range(49, 45, -1):
            if time.time() > deadline - 20:
                break
            trials = max(20, 80 - (49 - r) * 10)
            res = try_rank_structured(r, max_trials=trials)
            if res:
                best_result = res
                break
    
    if best_result:
        return best_result
    
    # Fallback to rank 50
    return current_solution()
