"""Capacitated facility location: Portfolio greedy seeds + optimized local search with swap moves."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _greedy_seed(demands, capacities, open_cost, assign_cost, strategy='demand_desc'):
    """
    Generate an initial feasible assignment using a greedy strategy.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac, dtype=float)

    # Determine the order in which to place customers
    if strategy == 'demand_desc':
        # Place largest demands first to avoid getting stuck with large unplaceable items
        # Tie-break by total assignment cost (lower is better)
        total_cost = assign_cost.sum(axis=1)
        order = np.lexsort((total_cost, -demands))
    elif strategy == 'demand_asc':
        order = np.argsort(demands)
    elif strategy == 'cheapest_first':
        # Place customers with the lowest minimum assignment cost first
        min_costs = np.min(assign_cost, axis=1)
        order = np.argsort(min_costs)
    elif strategy == 'random':
        order = np.random.permutation(n_cust)
    else:
        order = np.lexsort((assign_cost.sum(axis=1), -demands))

    # Precompute sorted facility indices for each customer based on assignment cost
    # fac_order[i] is a permutation of [0..n_fac-1] sorted by assign_cost[i, :]
    fac_order = np.argsort(assign_cost, axis=1)

    for i in order:
        placed = False
        # Try facilities in order of increasing assignment cost
        for f in fac_order[i]:
            if load[f] + demands[i] <= capacities[f] + 1e-9:
                load[f] += demands[i]
                result[i] = f
                placed = True
                break
        if not placed:
            # Fallback: assign to the facility with the most remaining capacity
            # This should rarely happen if total capacity > total demand
            f = int(np.argmin(load - capacities)) # Max remaining capacity
            if load[f] + demands[i] > capacities[f] + 1e-9:
                # This is infeasible, but we must return something. 
                # In valid instances, this shouldn't happen. 
                # We pick the one with least violation or just argmin load.
                f = int(np.argmin(load))
            load[f] += demands[i]
            result[i] = f
            
    return result, load

def _objective(result, load, open_cost, assign_cost):
    """Calculate the total cost: open costs + assignment costs."""
    used = load > 1e-9
    open_part = open_cost[used].sum()
    assign_part = assign_cost[np.arange(len(result)), result].sum()
    return open_part + assign_part

def _local_search(demands, capacities, open_cost, assign_cost, result, load, deadline):
    """
    Improve the assignment using 1-move and 2-swap local search.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Precompute sorted facility indices for each customer
    fac_order = np.argsort(assign_cost, axis=1)
    
    cur_obj = _objective(result, load, open_cost, assign_cost)
    
    improved = True
    no_improve_pass = 0
    max_no_improve = 2 # Stop if 2 full passes yield no improvement
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_pass = 0
        
        # --- Phase 1: Single Reassignment ---
        # For each customer, check if moving to another facility reduces cost
        for i in range(n_cust):
            if time.time() >= deadline:
                break
                
            f_old = result[i]
            d_i = demands[i]
            
            # Current assignment cost
            base_assign_cost = assign_cost[i, f_old]
            
            best_gain = 0.0
            best_f = -1
            
            # Check all other facilities
            # We iterate through fac_order[i] which is sorted by cost, 
            # but we must check capacity and open cost deltas.
            for f_new in fac_order[i]:
                if f_new == f_old:
                    continue
                
                # Capacity check
                if load[f_new] + d_i > capacities[f_new] + 1e-9:
                    continue
                
                # Calculate delta in assignment cost
                da = assign_cost[i, f_new] - base_assign_cost
                
                # Calculate delta in open cost
                do = 0.0
                # If f_old becomes empty, we save open_cost[f_old]
                if load[f_old] - d_i <= 1e-9:
                    do -= open_cost[f_old]
                # If f_new was empty, we pay open_cost[f_new]
                if load[f_new] <= 1e-9:
                    do += open_cost[f_new]
                
                # Total cost change
                delta_cost = da + do
                gain = -delta_cost # We want to minimize cost, so gain is negative delta
                
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_f = f_new
            
            if best_f != -1:
                # Apply the move
                load[f_old] -= d_i
                load[best_f] += d_i
                result[i] = best_f
                cur_obj -= best_gain
                improved = True
        
        if time.time() >= deadline:
            break
            
        # --- Phase 2: Swap Moves ---
        # Swap customers i and j if they are in different facilities
        # This allows us to fix cases where single moves are blocked by capacity
        # or open cost structures.
        for i in range(n_cust):
            if time.time() >= deadline:
                break
            
            f_i = result[i]
            d_i = demands[i]
            
            for j in range(i + 1, n_cust):
                if time.time() >= deadline:
                    break
                
                f_j = result[j]
                if f_i == f_j:
                    continue
                
                d_j = demands[j]
                
                # Capacity checks after swap
                # f_i loses d_i, gains d_j
                # f_j loses d_j, gains d_i
                new_load_i = load[f_i] - d_i + d_j
                new_load_j = load[f_j] - d_j + d_i
                
                if new_load_i > capacities[f_i] + 1e-9 or new_load_j > capacities[f_j] + 1e-9:
                    continue
                
                # Cost delta
                # Assignment cost change
                da = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                
                # Open cost change
                do = 0.0
                # Check if f_i becomes empty
                if new_load_i <= 1e-9 and load[f_i] > 1e-9:
                    do -= open_cost[f_i]
                # Check if f_j becomes empty
                if new_load_j <= 1e-9 and load[f_j] > 1e-9:
                    do -= open_cost[f_j]
                # Note: Facilities don't become "newly opened" in a swap if they had load before,
                # and they don't close if they still have load.
                # The only open cost changes are if a facility goes from >0 to 0.
                
                delta_cost = da + do
                gain = -delta_cost
                
                if gain > 1e-12:
                    # Apply swap
                    load[f_i] = new_load_i
                    load[f_j] = new_load_j
                    result[i], result[j] = f_j, f_i
                    cur_obj -= gain
                    improved = True
        
        if not improved:
            no_improve_pass += 1
            if no_improve_pass >= max_no_improve:
                break
        else:
            no_improve_pass = 0
            
    return result, load, cur_obj

def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign customers to facilities minimizing
    open_cost + assign_cost, respecting capacities.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Portfolio of initializations
    # We use different strategies to get diverse starting points
    strategies = ['demand_desc', 'cheapest_first', 'random', 'demand_asc']
    
    best_cost = float('inf')
    best_result = None
    best_load = None
    
    # Run multiple iterations of the portfolio
    # We don't know how fast the local search is, so we loop until time runs out
    iteration = 0
    while time.time() < deadline - 10.0: # Leave 10s for final validation/repair
        # Select a strategy
        strategy = strategies[iteration % len(strategies)]
        
        # For 'random', we want a different random seed each time
        if strategy == 'random':
            np.random.seed(int(time.time() * 1000) % (2**32))
            
        result, load = _greedy_seed(demands, capacities, open_cost, assign_cost, strategy)
        
        # Quick check for feasibility (should be guaranteed by greedy)
        if np.any(load > capacities + 1e-9):
            # Repair if necessary (rare)
            for f in range(n_fac):
                if load[f] > capacities[f] + 1e-9:
                    excess = load[f] - capacities[f]
                    custs_at_f = np.where(result == f)[0]
                    # Move largest demands first to free up space
                    custs_at_f = custs_at_f[np.argsort(-demands[custs_at_f])]
                    for i in custs_at_f:
                        if excess <= 1e-9:
                            break
                        # Find a feasible facility
                        for f2 in np.argsort(assign_cost[i]):
                            if f2 == f:
                                continue
                            if load[f2] + demands[i] <= capacities[f2] + 1e-9:
                                load[f] -= demands[i]
                                load[f2] += demands[i]
                                result[i] = f2
                                excess -= demands[i]
                                break
        
        # Local search
        # Allocate remaining time for this iteration, but leave buffer
        iter_deadline = min(deadline - 5.0, start_time + 20.0 + iteration * 5.0)
        # Actually, just use a fixed slice or the main deadline?
        # Better to let it run until the main deadline is close.
        # But if we have many iterations, we should cap each one.
        # Let's cap each local search to 15 seconds or until no improvement.
        ls_deadline = min(deadline - 2.0, time.time() + 15.0)
        
        result, load, current_cost = _local_search(
            demands, capacities, open_cost, assign_cost, 
            result, load, ls_deadline
        )
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_result = result.copy()
            best_load = load.copy()
            
        iteration += 1
        
        # If we've done a few good iterations and time is tight, break
        if iteration > 5 and time.time() > deadline - 30.0:
            break

    if best_result is None:
        # Fallback
        best_result, best_load = _greedy_seed(demands, capacities, open_cost, assign_cost, 'demand_desc')
        # Do a quick local search if time permits
        if time.time() < deadline - 5.0:
            best_result, best_load, _ = _local_search(
                demands, capacities, open_cost, assign_cost, 
                best_result, best_load, deadline - 2.0
            )

    # Final validation and repair
    result = best_result
    load_check = np.zeros(n_fac)
    for i in range(n_cust):
        f = result[i]
        if f < 0 or f >= n_fac:
            f = int(np.argmin(load_check))
            result[i] = f
        load_check[f] += demands[i]

    # Repair any capacity violations
    for f in range(n_fac):
        if load_check[f] > capacities[f] + 1e-9:
            excess = load_check[f] - capacities[f]
            custs_at_f = np.where(result == f)[0]
            custs_at_f = custs_at_f[np.argsort(-demands[custs_at_f])]
            for i in custs_at_f:
                if excess <= 1e-9:
                    break
                for f2 in np.argsort(assign_cost[i]):
                    if f2 == f:
                        continue
                    if load_check[f2] + demands[i] <= capacities[f2] + 1e-9:
                        load_check[f] -= demands[i]
                        load_check[f2] += demands[i]
                        result[i] = f2
                        excess -= demands[i]
                        break

    return result.astype(int)

# EVOLVE-BLOCK-END
