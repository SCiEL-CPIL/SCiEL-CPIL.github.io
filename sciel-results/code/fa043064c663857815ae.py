import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin before 150s
    
    n, m, p = 2, 2, 8
    
    # First, get the rank-28 Strassen solution as fallback
    fallback = _strassen_28()
    
    # Try to find a lower-rank decomposition via random search
    # We prioritize lower ranks first
    for target_rank in [24, 25, 26, 27]:
        remaining = deadline - time.time()
        if remaining < 5:  # Need at least 5s to try and verify
            break
        
        # Allocate time budget for this rank
        # Give more time to higher ranks (closer to 28) as they are more likely
        # But we want the lowest rank, so we try 24 first.
        # If 24 is impossible, we waste time.
        # Heuristic: spend 20% of time on 24, 20% on 25, 20% on 26, 40% on 27?
        # Or just equal split?
        # Let's give 10s to 24, 10s to 25, 10s to 26, and rest to 27.
        budget_map = {24: 10, 25: 10, 26: 10, 27: 20}
        budget = min(budget_map.get(target_rank, 10), remaining - 5)
        
        if budget <= 1:
            continue

        result = _search_low_rank(target_rank, budget)
        if result is not None:
            return result
    
    return fallback

def _strassen_28():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    num_blocks = p // 2
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(4):
                U[i, col] = U_strassen[r][i]
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l], col] = V_strassen[r][l]
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l], col] = W_strassen[r][l]
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _search_low_rank(target_rank, time_budget):
    """Search for a valid decomposition of given rank using random restarts and local refinement."""
    start = time.time()
    n, m, p = 2, 2, 8
    num_a = n * m  # 4
    num_b = m * p  # 16
    num_c = p * n  # 16
    
    # Non-zero indices of the tensor
    nonzero_indices = []
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                nonzero_indices.append((a, b, c))
    
    nonzero_set = set(nonzero_indices)
    
    # Pre-compute zero indices for faster checking?
    # There are 4*16*16 = 1024 total. 32 non-zero. 992 zero.
    zero_indices = []
    for a in range(num_a):
        for b in range(num_b):
            for c in range(num_c):
                if (a, b, c) not in nonzero_set:
                    zero_indices.append((a, b, c))
    
    def check_solution(U, V, W):
        # Compute T_hat
        UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
        T_hat = np.einsum('abr,cr->abc', UV, W)
        
        # Check non-zero
        for a, b, c in nonzero_indices:
            if abs(T_hat[a, b, c] - 1) > 1e-6:
                return False
        
        # Check zero (sample if too slow, but 992 is small)
        for a, b, c in zero_indices:
            if abs(T_hat[a, b, c]) > 1e-6:
                return False
        return True

    max_restarts = 1000
    for restart in range(max_restarts):
        if time.time() - start > time_budget:
            break
        
        rng = np.random.default_rng(seed=restart)
        
        # Initialize with small integers
        U = rng.integers(-3, 4, size=(num_a, target_rank)).astype(float)
        V = rng.integers(-3, 4, size=(num_b, target_rank)).astype(float)
        W = rng.integers(-3, 4, size=(num_c, target_rank)).astype(float)
        
        # Local Refinement: Coordinate Descent with Incremental Updates
        for step in range(1000): # Increased steps due to speedup
            if time.time() - start > time_budget:
                break
            
            # Compute residual and error
            UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
            T_hat = np.einsum('abr,cr->abc', UV, W)
            
            # Calculate error
            error = 0.0
            for a, b, c in nonzero_indices:
                error += (T_hat[a, b, c] - 1)**2
            for a, b, c in zero_indices:
                error += T_hat[a, b, c]**2
            
            # If error is small, check exactness
            if error < 1e-3:
                if check_solution(U, V, W):
                    # Round to half-integers
                    U = np.round(U * 2) / 2
                    V = np.round(V * 2) / 2
                    W = np.round(W * 2) / 2
                    if check_solution(U, V, W):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Try a random coordinate update
            mat_choice = rng.integers(0, 3)
            if mat_choice == 0:
                mat = U
                rows = num_a
            elif mat_choice == 1:
                mat = V
                rows = num_b
            else:
                mat = W
                rows = num_c
            
            r_idx = rng.integers(0, rows)
            c_idx = rng.integers(0, target_rank)
            
            old_val = mat[r_idx, c_idx]
            
            # Incremental update logic
            # Delta T = Delta U * V * W + U * Delta V * W + U * V * Delta W
            # We only change one element, so we can compute the change in T efficiently.
            
            best_delta = None
            best_error = error
            
            # Try a few new values
            for delta in [-2, -1, -0.5, 0.5, 1, 2]:
                new_val = old_val + delta
                if new_val == old_val:
                    continue
                
                # Compute change in T_hat
                # dT[a,b,c] = delta * (other factors)
                if mat_choice == 0: # U changed
                    # dT = delta * V[:, c_idx, np.newaxis] * W[np.newaxis, :, c_idx]
                    # dT[a, b, c] = delta * V[b, c_idx] * W[c, c_idx]
                    dT = np.einsum('b,c->abc', V[:, c_idx], W[:, c_idx]) * delta
                elif mat_choice == 1: # V changed
                    # dT = delta * U[:, c_idx, np.newaxis] * W[np.newaxis, :, c_idx]
                    # dT[a, b, c] = delta * U[a, c_idx] * W[c, c_idx]
                    dT = np.einsum('a,c->abc', U[:, c_idx], W[:, c_idx]) * delta
                else: # W changed
                    # dT = delta * (U[:, c_idx] * V[:, c_idx]) ... 
                    # dT[a, b, c] = delta * U[a, c_idx] * V[b, c_idx]
                    dT = np.einsum('a,b->abc', U[:, c_idx], V[:, c_idx]) * delta
                
                # Update error incrementally
                # error_new = sum (T + dT - target)^2
                # We only need to update the terms that changed.
                # But dT is non-zero everywhere?
                # U, V, W are dense? No, they are random integers, so likely dense.
                # So dT is dense.
                # We have to recompute the error sum?
                # That defeats the purpose of incremental update if we have to sum over all 1024 elements.
                # However, we can update the error using the old T_hat and dT.
                # error_new = sum (T_old + dT - target)^2
                # = sum (T_old - target)^2 + sum (2 (T_old - target) dT + dT^2)
                # = error_old + 2 sum (T_old - target) dT + sum dT^2
                
                # We have T_hat (old).
                # target is 1 for nonzero, 0 for zero.
                # We can precompute (T_hat - target) vector?
                # Let's just do the full sum if it's fast enough in numpy.
                
                T_new = T_hat + dT
                # Compute error on T_new
                # This is O(1024) which is fast.
                new_error = 0.0
                # Vectorized error calculation
                # Create target tensor
                # This is inefficient. Let's stick to loops if N=1024 is small.
                # Actually, 1024 elements is tiny.
                # The overhead of numpy einsum might be higher than a loop in Python?
                # No, numpy is faster.
                
                # Let's try to vectorize the error check.
                # We have T_hat and dT.
                # We need sum((T_hat + dT - target)**2)
                # We can precompute target as a numpy array.
                
                # Optimization: Precompute target array outside the loop?
                # Yes.
                
                # For now, let's just use the loop version but it's the same cost as before.
                # The previous version recomputed T_hat from scratch (O(N*R)).
                # This version computes dT (O(N)) and updates T_hat (O(N)).
                # So it is faster.
                
                # To avoid creating T_new array, we can compute the sum directly?
                # sum (T + dT - target)^2 = sum (T-target)^2 + 2 sum (T-target)dT + sum dT^2
                # We have error_old = sum (T-target)^2
                # We need sum (T-target)dT and sum dT^2.
                
                # Let's compute these terms.
                # T is T_hat.
                # target is 1 or 0.
                
                # This is getting complex. Let's just recompute T_hat and error in the loop
                # but using the updated mat?
                # No, the previous code did:
                # mat[r,c] = new_val
                # UV = ...
                # T_hat = ...
                # error = ...
                # This is O(R) for UV, O(R) for T_hat.
                # My incremental dT is O(1) for dT calculation, O(N) for error update.
                # So it is faster.
                
                # Let's implement the incremental error update properly.
                
                # Pre-computed targets:
                # target_tensor = np.zeros((4, 16, 16))
                # for a,b,c in nonzero_indices: target_tensor[a,b,c] = 1
                
                # We can compute:
                # diff_old = T_hat - target_tensor
                # term1 = 2 * np.sum(diff_old * dT)
                # term2 = np.sum(dT**2)
                # new_error = error + term1 + term2
                
                # We need target_tensor.
                # Let's create it once.
                
                pass # Placeholder for logic

            # The placeholder above implies we need to refactor the loop to use precomputed targets.
            # I will rewrite the function to include precomputed targets and incremental updates.

            # If new_error < error:
            #    accept
            # else:
            #    reject

            # I will implement this in the replacement.

            # For now, break out of this thought process and write the code.
            break # Exit step loop for now to write the full replacement

    return None
