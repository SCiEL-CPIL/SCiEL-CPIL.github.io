import numpy as np
import time

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Random permutation
    p = np.random.permutation(n)
    
    # Precompute helper structures for fast delta calculation
    # We will update these inside the loop when p changes.
    
    def get_best_move(p, a, b, tabu_mask):
        """
        Finds the best move (i, j) i < j such that (i, j) is not in tabu_mask.
        tabu_mask is a boolean matrix (n, n) where True means tabu.
        Returns (i, j, delta)
        """
        Bp = b[p, :] # (n, n), row i is b[p[i], :]
        
        # S(i, j) = dot(a[i] - a[j], Bp[j] - Bp[i])
        # = dot(a[i], Bp[j]) - dot(a[i], Bp[i]) - dot(a[j], Bp[j]) + dot(a[j], Bp[i])
        
        M1 = a @ Bp.T  # M1[i, j] = dot(a[i], Bp[j])
        M2 = Bp @ a.T  # M2[i, j] = dot(Bp[i], a[j]) = dot(a[j], Bp[i])
        
        diag_M1 = np.diag(M1) # dot(a[i], Bp[i])
        diag_M2 = np.diag(M2) # dot(a[j], Bp[j]) -- wait, M2[i, i] = dot(Bp[i], a[i]) = dot(a[i], Bp[i]) = diag_M1[i]
        # So diag_M1 and diag_M2 are identical.
        
        # S[i, j] = M1[i, j] - diag_M1[i] - diag_M1[j] + M2[i, j]
        S = M1 - diag_M1[:, None] - diag_M1[None, :] + M2
        
        # Correction Terms
        # t1 = -(a[i, i] - a[j, i]) * (Bp[j, i] - Bp[i, i])
        # Let A_ii = a[i, i], A_ij = a[i, j]
        # Let Bp_ii = Bp[i, i], Bp_ij = Bp[i, j]
        
        A_ii = np.diag(a)
        A_ij = a
        Bp_ii = np.diag(Bp)
        Bp_ij = Bp
        
        # t1[i, j] = -(A_ii[i] - A_ij[j, i]) * (Bp_ij[j, i] - Bp_ii[i])
        # Note: A_ij[j, i] is a[j, i].
        
        # We need to be careful with indices.
        # We want a matrix T where T[i, j] is the correction for swap (i, j).
        
        # t1
        term1_a = A_ii[:, None] - a.T # (i, j) -> A_ii[i] - a[j, i]
        term1_b = Bp.T - Bp_ii[:, None] # (i, j) -> Bp[j, i] - Bp_ii[i]
        T1 = -term1_a * term1_b
        
        # t2
        # t2[i, j] = -(a[i, j] - a[j, j]) * (Bp[j, j] - Bp[i, j])
        term2_a = a - A_ii[None, :] # (i, j) -> a[i, j] - A_ii[j]
        term2_b = Bp_ii[None, :] - Bp # (i, j) -> Bp_ii[j] - Bp[i, j]
        T2 = -term2_a * term2_b
        
        # t3
        # t3[i, j] = (a[i, j] - a[j, i]) * (Bp[j, i] - Bp[i, j])
        term3_a = a - a.T # (i, j) -> a[i, j] - a[j, i]
        term3_b = Bp.T - Bp # (i, j) -> Bp[j, i] - Bp[i, j]
        T3 = term3_a * term3_b
        
        # t4
        # t4[i, j] = (a[i, i] - a[j, j]) * (Bp[j, j] - Bp[i, i])
        term4_a = A_ii[:, None] - A_ii[None, :] # (i, j) -> A_ii[i] - A_ii[j]
        term4_b = Bp_ii[None, :] - Bp_ii[:, None] # (i, j) -> Bp_ii[j] - Bp_ii[i]
        T4 = term4_a * term4_b
        
        Delta = S + T1 + T2 + T3 + T4
        
        # Set diagonal to inf (no self-swap)
        np.fill_diagonal(Delta, np.inf)
        
        # Mask tabu moves
        # tabu_mask is symmetric? Yes, (i, j) same as (j, i).
        # We only consider i < j.
        # Create a mask for upper triangle.
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        # Apply tabu mask
        Delta = np.where(tabu_mask, np.inf, Delta)
        Delta = np.where(mask, Delta, np.inf)
        
        # Find min
        min_val = np.min(Delta)
        if np.isinf(min_val):
            return -1, -1, np.inf
            
        idx = np.unravel_index(np.argmin(Delta), Delta.shape)
        return idx[0], idx[1], min_val

    # 2. Initial Local Search (Steepest Descent)
    # Use the vectorized best move finder with empty tabu list
    tabu_mask = np.zeros((n, n), dtype=bool)
    cost = np.sum(a * b[np.ix_(p, p)])
    
    start_time = time.time()
    time_limit = 115
    
    # Quick local search to start
    while time.time() - start_time < 5: # Limit initial LS to 5s
        i, j, d = get_best_move(p, a, b, tabu_mask)
        if i == -1 or d >= 0:
            break
        p[i], p[j] = p[j], p[i]
        cost += d

    # 3. Tabu Search
    tabu_list = {} # (i, j) -> expiration_iter
    iter_count = 0
    best_cost = cost
    best_p = p.copy()
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        # Update tabu mask
        # Remove expired
        for key in list(tabu_list.keys()):
            if tabu_list[key] < iter_count:
                del tabu_list[key]
                
        # Build mask
        tabu_mask = np.zeros((n, n), dtype=bool)
        for (i, j) in tabu_list.keys():
            tabu_mask[i, j] = True
            tabu_mask[j, i] = True
            
        i, j, d = get_best_move(p, a, b, tabu_mask)
        
        if i == -1:
            # No moves available? Break or random
            break
            
        # Perform move
        p[i], p[j] = p[j], p[i]
        cost += d
        
        # Update tabu
        # Store symmetric key
        key = (i, j) if i < j else (j, i)
        tabu_list[key] = iter_count + 10 # Tabu for 10 iterations
        
        # Update best
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            no_improve = 0
        else:
            no_improve += 1
            
        iter_count += 1
        
        # Optional: If no improvement for long, restart?
        # For now, just continue.
        
    return best_p.tolist()
