import numpy as np

def construct():
    N = 24
    M = 4096  # Grid size for approximation
    angles = 2 * np.pi * np.arange(M) / M
    k_idx = np.arange(N)
    Z = np.exp(1j * np.outer(angles, k_idx))
    
    def fast_max_norm(c):
        P = Z @ c
        return np.max(np.abs(P))
        
    def precise_max_norm(c):
        # c is a numpy array of shape (24,)
        # Find approximate peaks on the grid
        P = Z @ c
        abs_P = np.abs(P)
        
        # Get top 10 indices
        top_indices = np.argsort(abs_P)[-10:][::-1]
        
        refined_max = 0.0
        for idx in top_indices:
            # Golden section search for local max of |P(e^{i theta})|
            delta = 2 * np.pi / M
            a = angles[idx] - delta
            b = angles[idx] + delta
            
            def f(theta):
                z = np.exp(1j * theta)
                return np.abs(np.dot(c, z ** k_idx))
            
            gr = (np.sqrt(5) - 1) / 2
            x1 = b - gr * (b - a)
            x2 = a + gr * (b - a)
            f1 = f(x1)
            f2 = f(x2)
            
            for _ in range(40):
                if f1 < f2:
                    a = x1
                    x1 = x2
                    f1 = f2
                    x2 = a + gr * (b - a)
                    f2 = f(x2)
                else:
                    b = x2
                    x2 = x1
                    f2 = f1
                    x1 = b - gr * (b - a)
                    f1 = f(x1)
                    
            peak_val = max(f1, f2)
            if peak_val > refined_max:
                refined_max = peak_val
                
        return refined_max

    # Tabu Search Parameters
    tabu_len = 15
    max_iters = 2000
    num_restarts = 50
    
    best_global_score = np.inf
    best_global_coeffs = None
    
    # Generate Golay-Rudin-Shapiro for initial guess
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    grs_coeffs = np.array(p[:N], dtype=float)
    
    for r in range(num_restarts):
        if r == 0:
            current = grs_coeffs.copy()
        else:
            current = np.random.choice([-1, 1], size=N).astype(float)
            
        current_cost = fast_max_norm(current)
        best_local_cost = current_cost
        best_local_coeffs = current.copy()
        
        tabu_list = [0] * tabu_len
        tabu_iter = 0
        no_improve_count = 0
        
        for it in range(max_iters):
            # Evaluate all 24 neighbors
            # We can vectorize the neighbor evaluation to some extent, 
            # but N=24 is small, so a loop is fine and clearer.
            # However, for speed, let's try to vectorize the cost calculation.
            
            # V = Z @ current
            V = Z @ current
            
            # For each k in 0..N-1:
            #   neighbor_k = current with c_k flipped
            #   P_k = V - 2 * c_k * Z[:, k]
            #   cost_k = max(abs(P_k))
            
            # We can compute all P_k at once?
            # P_k[m] = V[m] - 2 * current[k] * Z[m, k]
            # Let's create a matrix of shape (N, M)
            # term = 2 * current[:, None] * Z.T  # Shape (N, M)
            # neighbors_P = V[None, :] - term    # Shape (N, M)
            # neighbor_costs = np.max(np.abs(neighbors_P), axis=1) # Shape (N,)
            
            term = 2 * current[:, np.newaxis] * Z.T
            neighbors_P = V[np.newaxis, :] - term
            neighbor_costs = np.max(np.abs(neighbors_P), axis=1)
            
            # Apply tabu restrictions
            # tabu_list contains indices that are forbidden to flip for the next tabu_len iterations?
            # Standard Tabu: The move that was just made is tabu for the next L iterations.
            # Here, a "move" is flipping index k.
            # We need to know which indices were recently flipped.
            
            # Let's maintain a counter for each index indicating when it becomes non-tabu.
            # Or simpler: a list of (index, expiration_iter).
            # For efficiency, let's use an array `tabu_expiry` of size N.
            # If tabu_expiry[k] > it, then flipping k is tabu.
            
            # We need to initialize tabu_expiry
            if it == 0:
                tabu_expiry = np.zeros(N, dtype=int)
            
            # Mask for tabu moves
            is_tabu = tabu_expiry > it
            
            # If all moves are tabu (rare), or if we allow aspiration:
            # Aspiration: allow tabu move if it improves best_local_cost
            
            # Find best non-tabu move
            valid_costs = neighbor_costs.copy()
            valid_costs[is_tabu] = np.inf
            
            best_k = np.argmin(valid_costs)
            best_new_cost = neighbor_costs[best_k]
            
            # Check aspiration: if best_k was tabu, is it better than best_local?
            if is_tabu[best_k]:
                # If the best move is tabu, check if it's better than the current best local
                # Actually, we should pick the best move overall, and if it's tabu, 
                # only accept it if it improves best_local_cost.
                # Otherwise, pick the best non-tabu move.
                
                # Let's find the best non-tabu move index
                non_tabu_indices = np.where(~is_tabu)[0]
                if len(non_tabu_indices) > 0:
                    best_non_tabu_k = non_tabu_indices[np.argmin(neighbor_costs[non_tabu_indices])]
                    best_non_tabu_cost = neighbor_costs[best_non_tabu_k]
                    
                    if best_new_cost < best_local_cost:
                        # Aspiration: accept tabu move
                        chosen_k = best_k
                        chosen_cost = best_new_cost
                    else:
                        # Pick best non-tabu
                        chosen_k = best_non_tabu_k
                        chosen_cost = best_non_tabu_cost
                else:
                    # All moves are tabu (should not happen with L < N usually, or if we reset)
                    # Just pick the best overall
                    chosen_k = best_k
                    chosen_cost = best_new_cost
            else:
                chosen_k = best_k
                chosen_cost = best_new_cost
            
            # Apply move
            current[chosen_k] *= -1
            current_cost = chosen_cost
            
            # Update tabu list
            tabu_expiry[chosen_k] = it + tabu_len
            
            # Update best local
            if current_cost < best_local_cost:
                best_local_cost = current_cost
                best_local_coeffs = current.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Early break if stuck for too long?
            if no_improve_count > 100:
                break
                
        # After tabu search, refine the best local solution
        final_score = precise_max_norm(best_local_coeffs)
        
        if final_score < best_global_score:
            best_global_score = final_score
            best_global_coeffs = best_local_coeffs.copy()
            
        # Early stopping if we hit the known optimum
        if best_global_score < 1.2583759:
            break
            
    return best_global_coeffs.astype(int).tolist()
