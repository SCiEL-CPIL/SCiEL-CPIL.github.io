import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities to minimize total cost.
    Uses multi-start greedy initialization followed by local search (1-moves and 2-swaps).
    """
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Deadline guard: 145 seconds to be safe within the 150s limit
    deadline = time.time() + 145.0
    
    def evaluate(result):
        """
        Calculates the total cost of an assignment.
        Returns inf if infeasible.
        """
        load = np.zeros(n_facilities)
        for i, f in enumerate(result):
            load[f] += demands[i]
        
        # Check capacity feasibility
        if np.any(load > capacities + 1e-9):
            return float('inf')
        
        # Calculate open cost
        open_mask = load > 1e-9
        open_sum = np.sum(open_cost[open_mask])
        
        # Calculate assignment cost
        idx = np.arange(n_customers)
        assign_sum = np.sum(assign_cost[idx, result])
        
        return open_sum + assign_sum

    def greedy_init(randomize=False, seed=None):
        """
        Greedy initialization.
        If randomize is True, picks from top-k facilities with weighted probability.
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        is_open = np.zeros(n_facilities, dtype=bool)
        
        # Sort customers by demand descending (Largest Demand First)
        order = np.argsort(-demands)
        
        for i in order:
            d = demands[i]
            # Feasible facilities: load[f] + d <= capacities[f]
            feasible = load + d <= capacities
            
            if not np.any(feasible):
                # Fallback: pick facility with most remaining capacity (should rarely happen)
                f = np.argmax(capacities - load)
                result[i] = f
                load[f] += d
                is_open[f] = True
                continue
            
            # Calculate marginal costs: assignment cost + open cost if closing/opening
            # Note: In greedy init, we don't know if facility is "open" in the final sense,
            # but we can approximate by adding open_cost if load is 0.
            marginal_costs = assign_cost[i].copy()
            # If facility is empty (load==0), add open cost
            empty_mask = load == 0
            marginal_costs[empty_mask] += open_cost[empty_mask]
            
            # Mask infeasible
            marginal_costs[~feasible] = np.inf
            
            if randomize:
                # Pick from top 5 feasible facilities
                # Sort indices by marginal cost
                sorted_indices = np.argsort(marginal_costs)
                # Filter to feasible ones (marginal cost < inf)
                feasible_indices = sorted_indices[marginal_costs[sorted_indices] < np.inf]
                
                if len(feasible_indices) == 0:
                    f = np.argmax(capacities - load)
                else:
                    top_k = feasible_indices[:min(5, len(feasible_indices))]
                    # Weighted random selection: lower cost -> higher weight
                    # Use inverse cost or softmax-like weights
                    costs = marginal_costs[top_k]
                    # Shift to avoid zero division and negative weights if costs are negative (not expected here)
                    weights = 1.0 / (costs + 1.0)
                    weights /= weights.sum()
                    f = top_k[rng.choice(len(top_k), p=weights)]
            else:
                f = np.argmin(marginal_costs)
            
            result[i] = f
            load[f] += d
            is_open[f] = True
            
        return result, load, is_open

    def local_search(result, load, is_open):
        """
        Local search with 1-moves and 2-swaps.
        Returns improved result, load, is_open, and cost.
        """
        current_cost = evaluate(result)
        if current_cost == float('inf'):
            return result, load, is_open, current_cost

        improved = True
        # Limit passes to avoid infinite loops, but rely on time check
        max_passes = 50
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # --- Pass 1: 1-Moves (Relocation) ---
            # Random order to break symmetry
            c_order = np.random.permutation(n_customers)
            
            for c in c_order:
                if time.time() > deadline - 10.0:
                    return result, load, is_open, current_cost
                
                f_old = result[c]
                d_c = demands[c]
                
                best_delta = 1e-9
                best_f_new = -1
                
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    
                    # Check capacity
                    if load[f_new] + d_c > capacities[f_new] + 1e-9:
                        continue
                    
                    # Calculate cost change
                    delta_assign = assign_cost[c, f_new] - assign_cost[c, f_old]
                    
                    delta_open = 0.0
                    # If f_old becomes empty
                    if load[f_old] == d_c:
                        delta_open -= open_cost[f_old]
                    # If f_new was empty
                    if load[f_new] == 0.0:
                        delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new
                        
                if best_f_new != -1:
                    # Apply move
                    load[f_old] -= d_c
                    if load[f_old] == 0.0:
                        is_open[f_old] = False
                    load[best_f_new] += d_c
                    if load[best_f_new] == d_c: # Was 0
                        is_open[best_f_new] = True
                    result[c] = best_f_new
                    current_cost += best_delta
                    improved = True

            # --- Pass 2: 2-Swaps (Interchange) ---
            # This is O(N^2 * M) but N=100, M=20 is manageable.
            # We limit the number of swap attempts per pass to save time if needed.
            # Here we do a full pass but break early if time is tight.
            
            c_order = np.random.permutation(n_customers)
            
            for i in range(n_customers):
                if time.time() > deadline - 10.0:
                    return result, load, is_open, current_cost
                
                c1 = c_order[i]
                f1 = result[c1]
                d1 = demands[c1]
                
                for j in range(i + 1, n_customers):
                    c2 = c_order[j]
                    f2 = result[c2]
                    d2 = demands[c2]
                    
                    if f1 == f2:
                        continue
                    
                    # Check capacity feasibility for swap
                    # f1 loses d1, gains d2
                    # f2 loses d2, gains d1
                    new_load_f1 = load[f1] - d1 + d2
                    new_load_f2 = load[f2] - d2 + d1
                    
                    if new_load_f1 > capacities[f1] + 1e-9:
                        continue
                    if new_load_f2 > capacities[f2] + 1e-9:
                        continue
                    
                    # Calculate cost change
                    delta_assign = (assign_cost[c1, f2] - assign_cost[c1, f1]) + \
                                   (assign_cost[c2, f1] - assign_cost[c2, f2])
                    
                    delta_open = 0.0
                    # Open cost changes only if a facility becomes empty or was empty and becomes open.
                    # Since f1 and f2 had customers, they were open.
                    # They can only become empty.
                    
                    if load[f1] == d1 and new_load_f1 == 0.0:
                        # f1 becomes empty (only if d2==0, which is not possible as demands >= 5)
                        # Wait, if load[f1] == d1, it means c1 was the only customer.
                        # After swap, load is d2. Since d2 >= 5, it's not empty.
                        # So f1 stays open.
                        pass
                    
                    if load[f2] == d2 and new_load_f2 == 0.0:
                        # Similarly, f2 stays open.
                        pass
                    
                    # Actually, in a swap between two non-empty facilities, 
                    # neither facility becomes empty unless one of the customers has demand 0.
                    # Since demands >= 5, open cost does NOT change in a valid swap 
                    # where both facilities remain non-empty.
                    # However, if one facility had ONLY c1 (load=d1) and we move c1 out, 
                    # it gains d2. Since d2 > 0, it remains open.
                    # So delta_open is always 0 for swaps between non-empty facilities?
                    # Let's double check.
                    # Old state: f1 open, f2 open.
                    # New state: f1 open (load >= d2 > 0), f2 open (load >= d1 > 0).
                    # Yes, delta_open = 0.
                    
                    total_delta = delta_assign
                    
                    if total_delta < -1e-9:
                        # Accept swap
                        result[c1] = f2
                        result[c2] = f1
                        load[f1] = new_load_f1
                        load[f2] = new_load_f2
                        current_cost += total_delta
                        improved = True
                        # Break inner loop to restart outer loop with new state?
                        # Or just continue? Continuing is faster but might miss other swaps.
                        # Breaking is safer for convergence speed in local search.
                        break
        
        return result, load, is_open, current_cost

    # Main Loop: Multi-Start
    best_result = None
    best_score = float('inf')
    
    # We will run several restarts.
    # First one deterministic, others randomized.
    num_restarts = 50 # Max potential restarts, limited by time
    
    for k in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
            
        # Initialize
        if k == 0:
            init_result, init_load, init_is_open = greedy_init(randomize=False, seed=0)
        else:
            # Use different seeds for randomization
            init_result, init_load, init_is_open = greedy_init(randomize=True, seed=k)
        
        # Local Search
        ls_result, ls_load, ls_is_open, ls_score = local_search(init_result, init_load, init_is_open)
        
        # Update best
        if ls_score < best_score:
            best_score = ls_score
            best_result = ls_result.copy()
            
        # Check if we've hit a global lower bound? Hard to know.
        # If we have a very good score, we might stop early?
        # For now, let it run until time is up or restarts exhausted.
        
        # If we found a score that is "very good" (e.g., close to theoretical lower bound), 
        # we could break. But without a bound, we just continue.
        
    if best_result is None:
        # Fallback if something went wrong
        return np.zeros(n_customers, dtype=int)
        
    return best_result.astype(int)

# EVOLVE-BLOCK-END
