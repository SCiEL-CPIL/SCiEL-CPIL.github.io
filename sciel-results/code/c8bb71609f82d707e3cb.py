"""Capacitated facility location: Lagrangian relaxation with local search polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_seed(demands, capacities, assign_cost):
    """Greedy assignment: largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac, dtype=float)
    
    # Sort customers by demand descending, tie-break by total assign cost ascending
    total_cost = assign_cost.sum(axis=1)
    order = np.lexsort((total_cost, -demands))
    
    # Pre-sort facilities for each customer by assign cost
    fac_order = np.argsort(assign_cost, axis=1)
    
    for i in order:
        d = demands[i]
        assigned = False
        for f in fac_order[i]:
            if load[f] + d <= capacities[f]:
                load[f] += d
                result[i] = f
                assigned = True
                break
        if not assigned:
            # Should not happen if total capacity >= total demand, but handle gracefully
            # Assign to facility with most remaining capacity
            remaining = capacities - load
            best_f = np.argmax(remaining)
            result[i] = best_f
            load[best_f] += d
    
    return result


def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost of an assignment."""
    n_cust = len(result)
    n_fac = len(capacities)
    
    # Check validity
    load = np.zeros(n_fac)
    for i in range(n_cust):
        f = result[i]
        if f < 0 or f >= n_fac:
            return np.inf
        load[f] += demands[i]
    
    for f in range(n_fac):
        if load[f] > capacities[f] + 1e-6:
            return np.inf
    
    # Open cost
    used = set(result.tolist())
    open_cost_sum = sum(open_cost[f] for f in used)
    
    # Assignment cost
    assign_cost_sum = sum(assign_cost[i, result[i]] for i in range(n_cust))
    
    return open_cost_sum + assign_cost_sum


def _local_search(result, demands, capacities, open_cost, assign_cost, max_iter=500, deadline=None):
    """Local search: try reassigning individual customers to reduce cost."""
    n_cust = len(result)
    n_fac = len(capacities)
    
    best_result = result.copy()
    best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)
    
    if np.isinf(best_cost):
        return best_result, best_cost
    
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[best_result[i]] += demands[i]
    
    used_fac = np.zeros(n_fac, dtype=bool)
    for i in range(n_cust):
        used_fac[best_result[i]] = True
    
    # Precompute facility order by cost for each customer
    fac_order = np.argsort(assign_cost, axis=1)
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iters += 1
        
        # Try each customer
        for i in range(n_cust):
            current_f = best_result[i]
            current_d = demands[i]
            
            # Try all other facilities
            for f in fac_order[i]:
                if f == current_f:
                    continue
                
                # Check capacity
                if load[f] + current_d > capacities[f]:
                    continue
                
                # Compute delta in cost
                # Remove customer i from current_f
                old_load_current = load[current_f]
                new_load_current = old_load_current - current_d
                old_load_f = load[f]
                new_load_f = old_load_f + current_d
                
                # Open cost change
                old_open = 0.0
                new_open = 0.0
                if new_load_current <= 0 and used_fac[current_f]:
                    # This facility becomes unused
                    old_open = open_cost[current_f]
                if old_load_f <= 0 and not used_fac[f]:
                    # This facility becomes used
                    new_open = open_cost[f]
                
                # Assignment cost change
                old_assign = assign_cost[i, current_f]
                new_assign = assign_cost[i, f]
                
                delta = (new_open - old_open) + (new_assign - old_assign)
                
                if delta < -1e-9:
                    # Accept the move
                    load[current_f] = new_load_current
                    load[f] = new_load_f
                    
                    if new_load_current <= 0:
                        used_fac[current_f] = False
                    if new_load_f > 0 and not used_fac[f]:
                        used_fac[f] = True
                    
                    best_result[i] = f
                    best_cost += delta
                    improved = True
                    break  # Move to next customer
    
    return best_result, best_cost


def _lagrangian_relaxation(demands, capacities, open_cost, assign_cost, 
                            max_dual_iters=50, deadline=None):
    """Lagrangian relaxation: use dual variables to price capacity constraints."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Start with a greedy seed
    result = _greedy_seed(demands, capacities, assign_cost)
    best_result = result.copy()
    best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)
    
    # Dual variables for capacity constraints
    dual = np.zeros(n_fac)
    
    for dual_iter in range(max_dual_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Solve Lagrangian subproblem: minimize sum_i (assign_cost[i,f] + dual[f]*demands[i]) 
        # subject to each customer assigned to exactly one facility
        # This is equivalent to finding the cheapest facility for each customer
        # with the modified cost
        
        modified_cost = assign_cost + np.outer(demands, dual)
        
        # For each customer, assign to the cheapest facility under modified cost
        lag_result = np.argmin(modified_cost, axis=1)
        
        # Check if this is feasible
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[lag_result[i]] += demands[i]
        
        feasible = True
        for f in range(n_fac):
            if load[f] > capacities[f] + 1e-6:
                feasible = False
                break
        
        if feasible:
            cost = _compute_cost(lag_result, demands, capacities, open_cost, assign_cost)
            if cost < best_cost:
                best_cost = cost
                best_result = lag_result.copy()
        else:
            # Update dual variables: increase dual for overloaded facilities, decrease for underloaded
            for f in range(n_fac):
                if load[f] > capacities[f]:
                    dual[f] += 0.5 * (load[f] - capacities[f]) / (n_cust + 1)
                elif load[f] < capacities[f] * 0.5:
                    dual[f] = max(0, dual[f] - 0.1)
        
        # Also try local search on the current best result
        if deadline is not None and time.time() > deadline - 5:
            break
        polished, polished_cost = _local_search(best_result, demands, capacities, 
                                                 open_cost, assign_cost, 
                                                 max_iter=100, deadline=deadline)
        if polished_cost < best_cost:
            best_cost = polished_cost
            best_result = polished
    
    return best_result, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    """Assign customers to facilities using Lagrangian relaxation with local search."""
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Deadline: 140 seconds (safety margin before 150s limit)
    start_time = time.time()
    deadline = start_time + 140
    
    # Phase 1: Greedy seed
    result = _greedy_seed(demands, capacities, assign_cost)
    best_result = result.copy()
    best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)
    
    # Phase 2: Local search polish
    if time.time() < deadline - 30:
        polished, polished_cost = _local_search(best_result, demands, capacities, 
                                                 open_cost, assign_cost, 
                                                 max_iter=1000, deadline=deadline)
        if polished_cost < best_cost:
            best_cost = polished_cost
            best_result = polished
    
    # Phase 3: Lagrangian relaxation if time permits
    if time.time() < deadline - 10:
        lag_result, lag_cost = _lagrangian_relaxation(demands, capacities, open_cost, 
                                                       assign_cost, 
                                                       max_dual_iters=30, 
                                                       deadline=deadline)
        if lag_cost < best_cost:
            best_cost = lag_cost
            best_result = lag_result
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 5:
        polished, polished_cost = _local_search(best_result, demands, capacities, 
                                                 open_cost, assign_cost, 
                                                 max_iter=200, deadline=deadline)
        if polished_cost < best_cost:
            best_cost = polished_cost
            best_result = polished
    
    # Ensure result is valid integer array
    result = np.asarray(best_result, dtype=int)
    
    # Final validity check
    load = np.zeros(n_fac)
    for i in range(n_cust):
        f = result[i]
        if f < 0 or f >= n_fac:
            # Fallback to greedy
            result = _greedy_seed(demands, capacities, assign_cost)
            break
        load[f] += demands[i]
    
    for f in range(n_fac):
        if load[f] > capacities[f] + 1e-6:
            # Fallback to greedy
            result = _greedy_seed(demands, capacities, assign_cost)
            break
    
    return result
# EVOLVE-BLOCK-END
