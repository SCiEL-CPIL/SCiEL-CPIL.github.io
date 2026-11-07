import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    best_result = None
    best_score = float('inf')
    
    def evaluate(result):
        """Calculate the objective function value."""
        load = np.zeros(n_facilities)
        # Vectorized load calculation is tricky with variable indices, so we use bincount
        # result contains facility indices for each customer
        load = np.bincount(result, weights=demands, minlength=n_facilities)
        
        if np.any(load > capacities + 1e-9):
            return float('inf')
        
        open_mask = load > 0
        open_sum = np.sum(open_cost[open_mask])
        # Vectorized assignment cost sum
        assign_sum = np.sum(assign_cost[np.arange(n_customers), result])
        
        return open_sum + assign_sum

    def greedy_init(rng, randomize=False):
        """
        Greedy initialization.
        Sorts customers by demand descending.
        Assigns each customer to the facility with minimum marginal cost 
        (assignment cost + open cost if facility is closed).
        """
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        is_open = np.zeros(n_facilities, dtype=bool)
        
        # Sort customers by demand descending to handle large demands first
        order = np.argsort(-demands)
        
        for i in order:
            d = demands[i]
            # Feasible facilities: load + d <= capacity
            feasible = load + d <= capacities
            
            if not np.any(feasible):
                # Fallback: assign to facility with most remaining capacity or lowest cost
                # This should rarely happen if capacities are sufficient
                f = np.argmin(assign_cost[i])
                result[i] = f
                load[f] += d
                is_open[f] = True
                continue
            
            # Calculate marginal costs
            # Cost = assign_cost[i, f] + (open_cost[f] if not is_open[f] else 0)
            marginal_costs = assign_cost[i].copy()
            closed_mask = ~is_open
            marginal_costs[closed_mask] += open_cost[closed_mask]
            
            # Mask infeasible facilities
            marginal_costs[~feasible] = np.inf
            
            if randomize:
                # Pick from top K with randomness to diversify
                # Find indices of smallest costs
                sorted_indices = np.argsort(marginal_costs)
                # Take top 3 or all feasible if fewer
                k = min(3, np.sum(feasible))
                top_k_indices = sorted_indices[:k]
                
                # Weighted random selection: lower cost -> higher weight
                weights = 1.0 / (marginal_costs[top_k_indices] + 1e-9)
                weights /= weights.sum()
                f = top_k_indices[rng.choice(len(top_k_indices), p=weights)]
            else:
                f = np.argmin(marginal_costs)
            
            result[i] = f
            load[f] += d
            if load[f] > 0:
                is_open[f] = True
                
        return result, load, is_open

    def local_search_1_move(result, load, is_open, max_passes=10):
        """
        Local search using 1-moves (relocation).
        Tries to move a customer to a different facility if it reduces cost.
        """
        improved = True
        passes = 0
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            # Shuffle customers to avoid bias
            # But for speed, we iterate sequentially. 
            # We can randomize the order of checks per pass.
            check_order = np.arange(n_customers)
            rng = np.random.RandomState()
            rng.shuffle(check_order)
            
            for i in check_order:
                if time.time() > deadline:
                    break
                
                f_old = result[i]
                d = demands[i]
                
                if f_old == -1: continue # Should not happen
                
                best_delta = 1e-9
                best_f_new = -1
                
                # Iterate over all other facilities
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    
                    # Check capacity
                    if load[f_new] + d > capacities[f_new] + 1e-9:
                        continue
                    
                    # Calculate delta in assignment cost
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    
                    # Calculate delta in open cost
                    delta_open = 0.0
                    
                    # If old facility becomes empty, we save open_cost[f_old]
                    if load[f_old] == d:
                        delta_open -= open_cost[f_old]
                        
                    # If new facility was empty, we incur open_cost[f_new]
                    if load[f_new] == 0:
                        delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new
                        
                if best_f_new != -1:
                    # Apply move
                    load[f_old] -= d
                    if load[f_old] == 0:
                        is_open[f_old] = False
                        
                    load[best_f_new] += d
                    if load[best_f_new] == d: # It was 0
                        is_open[best_f_new] = True
                        
                    result[i] = best_f_new
                    improved = True
                    
            if time.time() > deadline:
                break
                
        return result, load, is_open

    def lns_search(initial_result, initial_load, initial_is_open, rng):
        """
        Large Neighborhood Search.
        Repeatedly destroys and repairs a subset of customers.
        """
        result = initial_result.copy()
        load = initial_load.copy()
        is_open = initial_is_open.copy()
        current_score = evaluate(result)
        
        best_local_result = result.copy()
        best_local_score = current_score
        
        # SA parameters
        temp = 50.0
        cooling = 0.99
        min_temp = 1e-3
        
        max_iter = 2000 # Max iterations for LNS
        
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            # Destroy: Remove k random customers
            # k can be adaptive, e.g., 10% to 30% of customers
            k = rng.randint(5, 21) # 5 to 20 customers
            k = min(k, n_customers)
            
            # Select customers to remove
            # Prefer customers with high assignment cost or in overloaded facilities?
            # For now, random selection is robust.
            remove_indices = rng.choice(n_customers, size=k, replace=False)
            
            # Save their current assignments to restore if we don't accept
            old_assignments = result[remove_indices].copy()
            
            # Remove them
            for i in remove_indices:
                f = result[i]
                load[f] -= demands[i]
                if load[f] == 0:
                    is_open[f] = False
                result[i] = -1 # Mark as unassigned
            
            # Repair: Re-assign the removed customers
            # Sort removed customers by demand descending? Or random?
            # Random order can help break ties.
            repair_order = remove_indices.copy()
            rng.shuffle(repair_order)
            
            repair_failed = False
            for i in repair_order:
                d = demands[i]
                feasible = load + d <= capacities
                
                if not np.any(feasible):
                    repair_failed = True
                    break
                
                # Calculate marginal costs for feasible facilities
                marginal_costs = np.full(n_facilities, np.inf)
                feasible_indices = np.where(feasible)[0]
                
                # Only calculate for feasible ones to save time? 
                # n_facilities is small (20), so calculating all is fine.
                marginal_costs[feasible_indices] = assign_cost[i, feasible_indices]
                closed_mask = ~is_open
                marginal_costs[feasible_indices & closed_mask] += open_cost[feasible_indices & closed_mask]
                
                f_new = np.argmin(marginal_costs)
                
                result[i] = f_new
                load[f_new] += d
                is_open[f_new] = True
            
            if repair_failed:
                # Restore original state
                for idx, i in enumerate(remove_indices):
                    f = old_assignments[idx]
                    result[i] = f
                    load[f] += demands[i]
                    is_open[f] = True
                continue
            
            # Polish the repaired solution with a quick local search
            result, load, is_open = local_search_1_move(result, load, is_open, max_passes=2)
            
            new_score = evaluate(result)
            
            # Acceptance criterion
            if new_score < current_score:
                current_score = new_score
                best_local_result = result.copy()
                best_local_score = new_score
            else:
                # Simulated Annealing acceptance
                delta = new_score - current_score
                if temp > min_temp and rng.rand() < np.exp(-delta / temp):
                    current_score = new_score
                    # We do not update best_local_result here unless it's better, 
                    # but we keep the worse solution as current to explore.
            
            temp *= cooling
        
        return best_local_result, best_local_score

    # Main Execution Logic
    
    # 1. Deterministic Greedy Initialization
    rng_det = np.random.RandomState(42)
    res_det, load_det, is_open_det = greedy_init(rng_det, randomize=False)
    res_det, load_det, is_open_det = local_search_1_move(res_det, load_det, is_open_det, max_passes=5)
    score_det = evaluate(res_det)
    
    if score_det < best_score:
        best_score = score_det
        best_result = res_det.copy()
    
    # 2. LNS from Deterministic Start
    if time.time() < deadline - 10:
        rng_lns = np.random.RandomState(42)
        res_lns, score_lns = lns_search(res_det, load_det, is_open_det, rng_lns)
        if score_lns < best_score:
            best_score = score_lns
            best_result = res_lns.copy()
            
    # 3. Randomized Restarts with LNS
    # Use remaining time for diverse restarts
    restart_count = 0
    while time.time() < deadline - 5:
        seed = 100 + restart_count
        rng = np.random.RandomState(seed)
        
        # Randomized greedy init
        res_r, load_r, is_open_r = greedy_init(rng, randomize=True)
        
        # Quick local search
        res_r, load_r, is_open_r = local_search_1_move(res_r, load_r, is_open_r, max_passes=2)
        score_r = evaluate(res_r)
        
        if score_r < best_score:
            best_score = score_r
            best_result = res_r.copy()
            
        # Run LNS on this new start
        # Limit LNS time per restart to ensure we can do multiple restarts
        # We'll let lns_search handle its own loop limit, but we check time outside
        # To be safe, we can pass a max_iter based on remaining time, 
        # but lns_search has a fixed max_iter. 
        # For now, just run it. If it takes too long, the internal check will break.
        
        # To prevent one LNS from eating all time, we can't easily limit it from outside 
        # without modifying lns_search to take a deadline. 
        # However, lns_search checks `time.time() > deadline - 2` inside its loop.
        # So it will stop early if time is up.
        
        res_lns_r, score_lns_r = lns_search(res_r, load_r, is_open_r, rng)
        
        if score_lns_r < best_score:
            best_score = score_lns_r
            best_result = res_lns_r.copy()
            
        restart_count += 1
        
        # If we've done a few restarts and time is getting tight, stop
        if restart_count > 50:
            break

    return best_result.astype(int)
