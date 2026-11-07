import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    best_result = None
    best_cost = np.inf
    
    # Helper: Calculate total cost
    def calc_cost(res):
        if res is None:
            return np.inf
        load = np.zeros(n_fac)
        for i, f in enumerate(res):
            load[f] += demands[i]
        if np.any(load > capacities + 1e-9):
            return np.inf
        open_mask = (np.bincount(res, minlength=n_fac) > 0)
        return np.sum(open_cost[open_mask]) + np.sum(assign_cost[np.arange(n_cust), res])
    
    # Helper: Greedy Initialization
    def greedy_init():
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        # Sort customers by demand descending to place large ones first
        order = np.argsort(-demands)
        for i in order:
            # Find feasible facilities
            feasible = np.where(load + demands[i] <= capacities)[0]
            if len(feasible) == 0:
                # Should not happen if instance is feasible, but fallback
                res[i] = 0
                continue
            # Choose min cost among feasible
            costs = assign_cost[i, feasible]
            best_f_idx = feasible[np.argmin(costs)]
            res[i] = best_f_idx
            load[best_f_idx] += demands[i]
        return res

    # Helper: Vectorized Local Search
    def local_search(res, time_limit):
        res = res.astype(int).copy()
        load = np.zeros(n_fac)
        counts = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(res):
            load[f] += demands[i]
            counts[f] += 1
            
        current_cost = calc_cost(res)
        if np.isinf(current_cost):
            return res, current_cost
            
        start_ls = time.time()
        improved = True
        
        while improved and (time.time() - start_ls < time_limit):
            improved = False
            
            # 1. 1-Move (Relocation)
            for i in range(n_cust):
                if time.time() - start_ls > time_limit:
                    break
                curr_f = res[i]
                d_i = demands[i]
                
                # Calculate delta for moving i to all other facilities
                # Base delta: assign_cost[i, j] - assign_cost[i, curr_f]
                deltas = assign_cost[i] - assign_cost[i, curr_f]
                
                # Open cost adjustments
                # If curr_f becomes empty, we save open_cost[curr_f]
                if counts[curr_f] == 1:
                    deltas -= open_cost[curr_f]
                
                # If target j was empty, we pay open_cost[j]
                empty_mask = (counts == 0)
                deltas[empty_mask] += open_cost[empty_mask]
                
                # Feasibility: capacity
                # load[j] + d_i <= capacities[j]
                infeasible_mask = (load + d_i > capacities)
                deltas[infeasible_mask] = np.inf
                # Cannot stay in same facility
                deltas[curr_f] = np.inf
                
                best_delta = np.min(deltas)
                if best_delta < -1e-9:
                    new_f = np.argmin(deltas)
                    # Apply move
                    load[curr_f] -= d_i
                    load[new_f] += d_i
                    counts[curr_f] -= 1
                    counts[new_f] += 1
                    res[i] = new_f
                    current_cost += best_delta
                    improved = True
            
            if time.time() - start_ls > time_limit:
                break
                
            # 2. 2-Move (Swap)
            # Iterate i, vectorize over j
            for i in range(n_cust):
                if time.time() - start_ls > time_limit:
                    break
                f_i = res[i]
                d_i = demands[i]
                
                # Only consider j > i to avoid duplicates and self-swaps
                j_indices = np.arange(i + 1, n_cust)
                if len(j_indices) == 0:
                    continue
                    
                f_j = res[j_indices]
                d_j = demands[j_indices]
                
                # Delta cost
                # c[i, f_j] + c[j, f_i] - c[i, f_i] - c[j, f_j]
                deltas = assign_cost[i, f_j] + assign_cost[j_indices, f_i] - assign_cost[i, f_i] - assign_cost[j_indices, f_j]
                
                # Capacity checks
                # Facility f_i: loses d_i, gains d_j
                cap_f_i_ok = (load[f_i] - d_i + d_j <= capacities[f_i])
                # Facility f_j: loses d_j, gains d_i
                cap_f_j_ok = (load[f_j] - d_j + d_i <= capacities[f_j])
                
                feasible_mask = cap_f_i_ok & cap_f_j_ok
                deltas[~feasible_mask] = np.inf
                
                best_delta = np.min(deltas)
                if best_delta < -1e-9:
                    best_j_idx = np.argmin(deltas)
                    j = j_indices[best_j_idx]
                    
                    # Apply swap
                    f_j_val = res[j]
                    # f_i loses d_i, gains d_j
                    # f_j loses d_j, gains d_i
                    load[f_i] = load[f_i] - d_i + d_j
                    load[f_j_val] = load[f_j_val] - d_j + d_i
                    res[i] = f_j_val
                    res[j] = f_i
                    current_cost += best_delta
                    improved = True
                    
        return res, current_cost

    # Multi-start loop
    restarts = 0
    while time.time() < deadline - 5.0:
        # Generate initial solution
        init_res = greedy_init()
        
        # Determine time slice for local search
        # Use a fixed slice, e.g., 2-5 seconds, to allow multiple restarts
        # Or adaptive: if we have lots of time, do longer search.
        time_slice = 3.0
        if deadline - time.time() < time_slice:
            time_slice = deadline - time.time() - 1.0
            
        if time_slice < 0.5:
            break
            
        res, cost = local_search(init_res, time_slice)
        
        if cost < best_cost:
            best_cost = cost
            best_result = res.copy()
            
        restarts += 1
        
    # If no restarts completed, use greedy
    if best_result is None:
        best_result = greedy_init()
        
    # Final validation
    load_final = np.zeros(n_fac)
    for i, f in enumerate(best_result):
        if f < 0 or f >= n_fac:
            # Fallback to zero
            best_result = np.zeros(n_cust, dtype=int)
            break
        load_final[f] += demands[i]
        
    if np.any(load_final > capacities + 1e-9):
        # Fallback to greedy which is known to be feasible (or try to fix)
        # Greedy is feasible by construction if instance is feasible.
        best_result = greedy_init()

    return best_result
