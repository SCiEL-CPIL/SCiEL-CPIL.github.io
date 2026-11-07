"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # 1. Initial feasible assignment using Randomized Greedy (multiple attempts to find good start)
    best_init = None
    best_init_cost = np.inf
    
    # Try a few randomization seeds for the greedy step
    for trial in range(10):
        if time.time() > deadline:
            break
            
        # Add small noise to costs to diversify the greedy path
        noise = np.random.uniform(0, 5, size=assign_cost.shape)
        noisy_cost = assign_cost + noise
        
        # Sort customers by demand descending
        order = np.argsort(-demands)
        
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        valid = True
        
        for i in order:
            # Sort facilities by noisy cost
            fac_order = np.argsort(noisy_cost[i])
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                valid = False
                break
        
        if valid:
            # Compute cost
            open_mask = (load > 0)
            open_sum = np.sum(open_cost[open_mask])
            assign_sum = np.sum(assign_cost[np.arange(n_cust), result])
            cost = open_sum + assign_sum
            
            if cost < best_init_cost:
                best_init_cost = cost
                best_init = result.copy()
                best_init_load = load.copy()
                
    if best_init is None:
        # Fallback: strict greedy without noise
        order = np.argsort(-demands)
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        best_init = result
        best_init_load = load

    current_result = best_init
    current_load = best_init_load
    current_cost = best_init_cost
    
    # 2. Local Search: Vectorized Moves
    # We alternate between:
    # A. Reassignment: Move customer i to facility f if it reduces cost.
    #    Cost change: 
    #    - assign_cost[i, old] removed
    #    + assign_cost[i, new] added
    #    - If old facility becomes empty: -open_cost[old]
    #    - If new facility was empty: +open_cost[new]
    # B. Swap: Swap customers i and j if it reduces cost.
    
    # Precompute indices
    cust_indices = np.arange(n_cust)
    
    while time.time() < deadline:
        improved = False
        
        # --- Phase A: Reassignment ---
        # For each customer, check all facilities.
        # Vectorize: 
        # current_assign_costs = assign_cost[cust_indices, current_result]
        # new_assign_costs = assign_cost (shape n_cust x n_fac)
        
        # Calculate potential savings in assignment cost
        # savings_assign[i, f] = current_assign_cost[i] - assign_cost[i, f]
        
        # Open cost adjustments:
        # If we move i from old to new:
        # - If load[old] == demands[i] (old becomes empty): save open_cost[old]
        # - If load[new] == 0 (new becomes open): pay open_cost[new]
        
        # This is tricky to fully vectorize because load changes.
        # However, we can do a "best improvement" pass.
        
        # Let's try a simpler vectorized heuristic:
        # For each customer, find the best alternative facility.
        
        # 1. Identify for each customer the current cost
        curr_assign_costs = assign_cost[cust_indices, current_result]
        
        # 2. Calculate cost to move to each facility f
        #    base_cost = assign_cost[i, f]
        #    delta_open = 0
        #    if current_load[old] == demands[i]: delta_open -= open_cost[old]
        #    if current_load[f] == 0: delta_open += open_cost[f]
        
        # We can approximate the open cost impact using current loads.
        # Note: This is a "one-step" lookahead. It might be suboptimal but fast.
        
        # Vectorized check for all i, f
        # curr_loads = current_load (shape n_fac)
        # demands = demands (shape n_cust)
        
        # For each i, old = current_result[i]
        # Condition 1: old becomes empty -> load[old] == demands[i]
        old_becomes_empty = (current_load[current_result] == demands) # shape (n_cust,)
        
        # Condition 2: new facility f is empty -> load[f] == 0
        # This depends on f, so we need a matrix.
        new_facility_empty = (current_load[None, :] == 0) # shape (1, n_fac) -> broadcast to (n_cust, n_fac)
        
        # Delta Open Cost Matrix (n_cust, n_fac)
        # If moving i to f:
        # - If old_becomes_empty[i]: -open_cost[old]
        # - If new_facility_empty[f]: +open_cost[f]
        
        open_cost_old = open_cost[current_result] # shape (n_cust,)
        open_cost_new = open_cost # shape (n_fac,)
        
        delta_open = np.zeros((n_cust, n_fac))
        # Subtract open cost of old if it becomes empty
        delta_open[np.arange(n_cust), :] -= np.where(old_becomes_empty[:, None], open_cost_old[:, None], 0)
        # Add open cost of new if it is empty
        delta_open += np.where(new_facility_empty, open_cost_new[None, :], 0)
        
        # Total Cost Change Matrix
        # change[i, f] = (assign_cost[i, f] + delta_open[i, f]) - curr_assign_costs[i]
        # We want change < 0.
        
        # Also must respect capacity: current_load[f] + demands[i] <= capacities[f]
        # And f != current_result[i] (moving to same place is no-op, but cost change is 0, so it won't be < 0 unless numerical issues, but let's mask it)
        
        feasible_mask = (current_load[None, :] + demands[:, None] <= capacities[None, :])
        # Mask out current assignment
        current_mask = np.zeros((n_cust, n_fac), dtype=bool)
        current_mask[cust_indices, current_result] = True
        feasible_mask &= ~current_mask
        
        # Calculate total change
        # Note: delta_open calculation above assumes that ONLY this customer moves.
        # This is valid for a single move.
        
        total_change = assign_cost + delta_open - curr_assign_costs[:, None]
        
        # Find best move for each customer
        # Set infeasible to inf
        total_change_masked = np.where(feasible_mask, total_change, np.inf)
        
        # For each customer, find the min change and the corresponding facility
        min_change_idx = np.argmin(total_change_masked, axis=1)
        min_change_val = np.min(total_change_masked, axis=1)
        
        # Filter customers where min_change_val < 0
        improving_mask = min_change_val < -1e-9
        
        if np.any(improving_mask):
            # Apply these moves
            # We must be careful: if multiple customers move, loads change.
            # To be safe and fast, we can apply them in batches or one by one.
            # Given 100 customers, one by one is fast enough (100 * vector ops).
            
            # Get indices of customers to move
            move_indices = np.where(improving_mask)[0]
            
            # Shuffle to avoid bias? Or just process in order.
            # Processing in order might lead to cascading effects.
            # Let's process them.
            
            for i in move_indices:
                if time.time() > deadline:
                    break
                
                f_new = min_change_idx[i]
                f_old = current_result[i]
                
                # Check capacity again (since previous moves in this batch might have filled it)
                if current_load[f_new] + demands[i] <= capacities[f_new]:
                    # Check if cost actually improves (recalculate if needed? The delta_open was based on old loads)
                    # Let's just do the move and update loads.
                    # The cost improvement calculation was an estimate.
                    # To be robust, we can verify the cost change.
                    
                    old_load = current_load[f_old]
                    new_load = current_load[f_new]
                    
                    # Cost before
                    cost_before = assign_cost[i, f_old]
                    if old_load == demands[i]:
                        cost_before += open_cost[f_old]
                    
                    # Cost after
                    cost_after = assign_cost[i, f_new]
                    if new_load == 0:
                        cost_after += open_cost[f_new]
                    
                    if cost_after < cost_before:
                        # Perform move
                        current_load[f_old] -= demands[i]
                        current_load[f_new] += demands[i]
                        current_result[i] = f_new
                        current_cost -= (cost_before - cost_after)
                        improved = True

        # --- Phase B: Swaps ---
        # Swap customers i and j.
        # This is O(N^2). N=100 -> 10,000 pairs.
        # Vectorizing swaps is complex due to capacity constraints.
        # We can do a random sample of swaps or a full pass if time permits.
        
        # Let's do a full pass of random pairs or sorted pairs?
        # A full O(N^2) loop in Python is slow (10k iterations * overhead).
        # 10k iterations is fine in 150s.
        
        # Optimization: Only consider pairs where current costs are high or facilities are different.
        
        # Let's try a vectorized approach for swaps if possible, or a sampled loop.
        # Given the small N, a loop over pairs is acceptable if optimized.
        
        # We can vectorize the cost calculation for all pairs (i, j)
        # But capacity check is hard.
        
        # Let's do a random sampling of swaps to keep it fast and stochastic.
        # Or just skip swaps if reassignment is still improving?
        # Reassignment usually handles most improvements. Swaps help with capacity bottlenecks.
        
        # Let's implement a simple loop for swaps, but limit it to avoid timeout.
        # If we have time, do a few passes.
        
        # For now, let's stick to reassignment as it's vectorized and fast.
        # If no reassignment improved, we might be stuck.
        # Let's add a simple swap mechanism if time allows.
        
        if not improved:
            # Try some swaps
            # Sample 1000 random pairs
            if time.time() < deadline - 5:
                pairs_checked = 0
                # Use random pairs
                while pairs_checked < 1000 and time.time() < deadline - 5:
                    i, j = np.random.randint(0, n_cust, 2)
                    if i == j: continue
                    pairs_checked += 1
                    
                    f_i = current_result[i]
                    f_j = current_result[j]
                    
                    if f_i == f_j:
                        continue # No swap needed if same facility
                    
                    # Check capacity for swap
                    # i moves to f_j, j moves to f_i
                    # load[f_i] - d_i + d_j <= cap[f_i]
                    # load[f_j] - d_j + d_i <= cap[f_j]
                    
                    if (current_load[f_i] - demands[i] + demands[j] <= capacities[f_i]) and \
                       (current_load[f_j] - demands[j] + demands[i] <= capacities[f_j]):
                        
                        # Calculate cost change
                        # Remove i from f_i, j from f_j
                        # Add i to f_j, j to f_i
                        
                        # Open cost changes:
                        # f_i becomes empty if load[f_i] == demands[i]
                        # f_j becomes empty if load[f_j] == demands[j]
                        # f_i becomes open if load[f_i] - demands[i] + demands[j] == 0 (impossible if d_j>0)
                        # f_j becomes open if load[f_j] - demands[j] + demands[i] == 0 (impossible if d_i>0)
                        # Wait, if f_i was open only for i, and we move i out, it closes.
                        # If f_j was open only for j, and we move j out, it closes.
                        # But we are moving i IN to f_j and j IN to f_i.
                        
                        # So:
                        # If load[f_i] == demands[i]: f_i closes -> -open_cost[f_i]
                        # If load[f_j] == demands[j]: f_j closes -> -open_cost[f_j]
                        
                        # No new facilities opened because we are moving into existing ones?
                        # Yes, f_i and f_j are already open (since i and j are assigned to them).
                        
                        delta_open = 0
                        if current_load[f_i] == demands[i]:
                            delta_open -= open_cost[f_i]
                        if current_load[f_j] == demands[j]:
                            delta_open -= open_cost[f_j]
                        
                        delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                        
                        total_delta = delta_assign + delta_open
                        
                        if total_delta < -1e-9:
                            # Perform swap
                            current_load[f_i] -= demands[i]
                            current_load[f_i] += demands[j]
                            current_load[f_j] -= demands[j]
                            current_load[f_j] += demands[i]
                            
                            current_result[i] = f_j
                            current_result[j] = f_i
                            
                            current_cost += total_delta
                            improved = True
                            
                            # Break inner loop to re-evaluate? Or continue?
                            # Continuing is fine, loads are updated.
                            
        # If no improvement in either phase, we might be in a local optimum.
        # We could restart with a different initialization, but we already did that.
        # We can break if no improvement for a while?
        # For now, just loop until deadline.
        
    return current_result
# EVOLVE-BLOCK-END
