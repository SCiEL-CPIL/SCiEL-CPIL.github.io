"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 118.0
    
    # Precompute M = A @ B.T
    # M[i, k] = sum_j a[i, j] * b[k, j]
    # Cost of assigning i to k is M[i, k]
    M = a @ b.T
    
    # Lagrange multipliers for location constraints
    # lambda[k] is added to the cost of assigning any facility to location k
    lambdas = np.zeros(n)
    
    best_cost = np.inf
    best_p = np.arange(n) # Initial fallback
    
    # Subgradient parameters
    alpha = 1.0
    alpha_min = 0.01
    decay = 0.999
    
    # We will iterate until time limit
    # Each iteration is O(n^2) due to sorting/assignment
    iterations = 0
    while time.time() - start_time < time_limit:
        # 1. Compute adjusted costs
        # adjusted_cost[i, k] = M[i, k] + lambdas[k]
        # We don't need to store the full matrix, just find argmins
        
        # Find best and second best location for each facility i
        # We can use np.argpartition for speed, but n=150 is small enough for sort
        # However, we need to handle conflicts.
        
        # Strategy: 
        # 1. For each i, find k_best and k_second.
        # 2. Calculate regret = cost_best - cost_second.
        # 3. Sort facilities by regret descending.
        # 4. Assign greedily.
        
        # To avoid O(n^2 log n) sort every time if we can be smarter? 
        # n=150, n^2=22500. Sorting 150 items is trivial.
        
        best_locs = np.zeros(n, dtype=int)
        second_locs = np.zeros(n, dtype=int)
        best_costs = np.zeros(n)
        second_costs = np.zeros(n)
        
        # Compute adjusted costs row by row? Or vectorized?
        # adjusted = M + lambdas[None, :]
        adjusted = M + lambdas[np.newaxis, :]
        
        # Get indices of smallest elements
        # np.argpartition returns indices of k smallest elements
        # We want the 2 smallest
        part_indices = np.argpartition(adjusted, 1, axis=1)
        
        # Extract the two smallest values and their indices
        # part_indices[:, 0] is index of smallest, part_indices[:, 1] is index of second smallest
        # But they are not sorted relative to each other.
        
        idx1 = part_indices[:, 0]
        idx2 = part_indices[:, 1]
        
        val1 = adjusted[np.arange(n), idx1]
        val2 = adjusted[np.arange(n), idx2]
        
        # Ensure idx1 is the one with smaller value
        swap_mask = val1 > val2
        best_locs = idx1.copy()
        second_locs = idx2.copy()
        best_costs = val1.copy()
        second_costs = val2.copy()
        
        # Swap where needed
        best_locs[swap_mask], second_locs[swap_mask] = second_locs[swap_mask], best_locs[swap_mask]
        best_costs[swap_mask], second_costs[swap_mask] = second_costs[swap_mask], best_costs[swap_mask]
        
        # Calculate regret
        regret = best_costs - second_costs
        
        # Sort facilities by regret (highest first)
        order = np.argsort(-regret, kind='stable')
        
        # Greedy Assignment
        p = np.full(n, -1, dtype=int)
        loc_used = np.zeros(n, dtype=bool)
        
        for i in order:
            k1 = best_locs[i]
            k2 = second_locs[i]
            
            if not loc_used[k1]:
                p[i] = k1
                loc_used[k1] = True
            elif not loc_used[k2]:
                p[i] = k2
                loc_used[k2] = True
            else:
                # Both best and second best are taken.
                # Find the best available location for i
                # We need to scan all locations for i
                # To make this faster, we can pre-sort all locations for each i? 
                # That's expensive. 
                # Given n=150, scanning 150 items is fast.
                # Find min of adjusted[i, :] where loc_used is False
                available_mask = ~loc_used
                # If no locations available, break (should not happen as n facilities, n locs)
                if available_mask.sum() == 0:
                    break
                # Mask out used locations
                adj_i = adjusted[i].copy()
                adj_i[loc_used] = np.inf
                k_best_avail = np.argmin(adj_i)
                p[i] = k_best_avail
                loc_used[k_best_avail] = True
        
        # Check if valid permutation
        if np.all(p >= 0) and len(np.unique(p)) == n:
            # Calculate cost
            # cost = sum a[i, j] * b[p[i], p[j]]
            # Efficient calculation:
            # B_p = b[np.ix_(p, p)]
            # cost = np.sum(a * B_p)
            # This is O(n^2)
            
            B_p = b[np.ix_(p, p)]
            current_cost = int(np.sum(a * B_p))
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
            # Update Lagrange multipliers
            # Subgradient g_k = (number of facilities assigned to k) - 1
            # Since p is a permutation, number assigned to each k is exactly 1.
            # So g_k = 0 for all k?
            # Wait. The Lagrangian relaxation minimizes L(lambda) = sum_i min_k (M[i,k] + lambda[k]).
            # The subgradient is the vector of chosen locations' counts minus 1.
            # In our greedy repair, we forced a permutation. 
            # The "true" subgradient from the relaxed problem (before repair) is:
            # g_k = sum_i 1[best_loc(i) == k] - 1
            # We should use the subgradient from the *relaxed* solution (the argmins before conflict resolution)
            # to guide the lambda updates, as that reflects the pressure on locations.
            
            counts = np.zeros(n)
            np.add.at(counts, best_locs, 1)
            g = counts - 1.0
            
            # Step size alpha
            # Standard subgradient: alpha_k = alpha * (UB - L(lambda)) / ||g||^2
            # We don't have L(lambda) easily, but we can approximate or just use decaying alpha.
            # Simple decaying alpha is often robust enough for heuristics.
            
            lambdas += alpha * g
            
            # Decay alpha
            alpha *= decay
            if alpha < alpha_min:
                alpha = alpha_min
                # Occasionally randomize lambdas to escape local minima of the dual
                if iterations % 100 == 0:
                    lambdas = np.random.normal(0, 100, n)
        
        iterations += 1
        
        # Optional: Run a quick local search (2-opt) on best_p to polish
        # Since we are in a different family, adding a local search polish is acceptable 
        # as a "repair" or "improvement" phase, but the core generation is Lagrangian.
        # Let's add a simple 2-opt if time permits, but keep it simple.
        # Actually, the Lagrangian approach itself is the main driver.
        
        # To ensure we don't get stuck, let's add a small amount of noise to lambdas occasionally
        if iterations % 50 == 0:
            lambdas += np.random.normal(0, 10, n)

    # Final Polish: 2-opt local search on best_p
    # This is standard for QAP and ensures we are at a local optimum.
    p = best_p.copy()
    current_cost = best_cost
    
    # Simple 2-opt
    # We can speed this up by precomputing delta costs? 
    # For n=150, O(n^2) swaps per pass.
    # Passes: maybe 10-20.
    
    # Helper for delta cost
    def delta_cost(i, j, p, a, b):
        u = p[i]
        v = p[j]
        
        # Vectorized delta calculation
        b_v_p = b[v, p]
        b_u_p = b[u, p]
        b_p_v = b[p, v]
        b_p_u = b[p, u]
        
        diff_v_u_out = b_v_p - b_u_p
        diff_v_u_in = b_p_v - b_p_u
        
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        s1 = np.dot(a[i, mask], diff_v_u_out[mask])
        s2 = np.dot(a[mask, i], diff_v_u_in[mask])
        s3 = np.dot(a[j, mask], -diff_v_u_out[mask])
        s4 = np.dot(a[mask, j], -diff_v_u_in[mask])
        
        sum_part = s1 + s2 + s3 + s4
        
        cross_delta = a[i, j] * b[v, u] + a[j, i] * b[u, v] - (a[i, j] * b[u, v] + a[j, i] * b[v, u])
        
        return sum_part + cross_delta

    # Run local search for remaining time
    while time.time() - start_time < time_limit - 1.0:
        improved = False
        # Randomize order to avoid cycles? Or just sweep.
        # Sweep is deterministic.
        for i in range(n - 1):
            for j in range(i + 1, n):
                d = delta_cost(i, j, p, a, b)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += d
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    # Continue searching for other improvements in this pass
                    # (Steepest descent would break, first-improvement continues)
        if not improved:
            break

    return best_p.tolist()
# EVOLVE-BLOCK-END
