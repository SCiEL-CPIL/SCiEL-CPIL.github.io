import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Greedy Initialization
    # Sort customers by demand descending
    order = np.argsort(-demands)
    result = np.full(n_customers, -1, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in order:
        # Find cheapest feasible facility
        # We can sort facilities by assign_cost[i]
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # Should not happen if total capacity >= total demand
            # If it does, assign to least loaded feasible? Or raise error?
            # Task implies feasible instances.
            pass
            
    # 2. Local Search
    # Precompute some things for speed?
    # We need to track load and open status.
    
    def get_cost(res, load):
        fixed = 0.0
        for f in range(n_facilities):
            if load[f] > 0:
                fixed += open_cost[f]
        var = np.sum([assign_cost[i, res[i]] for i in range(n_customers)])
        return fixed + var

    current_cost = get_cost(result, load)
    
    start_time = time.time()
    deadline = 140.0 # Safety margin for 150s limit
    
    # Precompute sorted facilities for each customer for faster 1-swap
    sorted_facs = np.argsort(assign_cost, axis=1)
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        
        # 1-swap moves
        # For each customer, try moving to every other facility
        for i in range(n_customers):
            f_old = result[i]
            # Try all other facilities in sorted order
            for f_new in sorted_facs[i]:
                if f_new == f_old:
                    continue
                
                # Check capacity
                if load[f_new] + demands[i] > capacities[f_new]:
                    continue
                
                # Calculate cost change
                delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                delta_fixed = 0.0
                
                # Check if f_old becomes empty
                if load[f_old] == demands[i]:
                    delta_fixed -= open_cost[f_old]
                
                # Check if f_new becomes open (was empty)
                if load[f_new] == 0:
                    delta_fixed += open_cost[f_new]
                
                if delta_assign + delta_fixed < -1e-9:
                    # Accept move
                    load[f_old] -= demands[i]
                    load[f_new] += demands[i]
                    result[i] = f_new
                    current_cost += delta_assign + delta_fixed
                    improved = True
                    break # Break inner loop
            if improved:
                # To be safe and simple, break out and restart the whole search iteration
                # to ensure consistency with updated loads.
                # This is a "first improvement" strategy with restart.
                break
        
        if improved:
            continue
            
        # 2-swap moves
        # For each pair of customers (i, j)
        for i in range(n_customers):
            f_i = result[i]
            for j in range(i + 1, n_customers):
                f_j = result[j]
                if f_i == f_j:
                    continue
                
                # Check capacity feasibility for swap
                # New load f_i: load[f_i] - demands[i] + demands[j]
                # New load f_j: load[f_j] - demands[j] + demands[i]
                new_load_fi = load[f_i] - demands[i] + demands[j]
                new_load_fj = load[f_j] - demands[j] + demands[i]
                
                if new_load_fi > capacities[f_i] or new_load_fj > capacities[f_j]:
                    continue
                
                # Calculate cost change
                delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                delta_fixed = 0.0
                
                # Check if f_i becomes empty
                if new_load_fi == 0:
                    delta_fixed -= open_cost[f_i]
                
                # Check if f_j becomes empty
                if new_load_fj == 0:
                    delta_fixed -= open_cost[f_j]
                
                # Note: f_i and f_j were open (load > 0) before swap.
                # They cannot become "open" if they were closed, because they had customers.
                # So we only check for closing.
                
                if delta_assign + delta_fixed < -1e-9:
                    # Accept swap
                    load[f_i] = new_load_fi
                    load[f_j] = new_load_fj
                    result[i] = f_j
                    result[j] = f_i
                    current_cost += delta_assign + delta_fixed
                    improved = True
                    break # Restart search iteration
            if improved:
                break
        
        # If no improvement found in 1-swap and 2-swap, we are at a local minimum
        # for these move types.
        
    return result
