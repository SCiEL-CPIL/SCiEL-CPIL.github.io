"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_assign(demands, capacities, open_cost, assign_cost):
    """Greedy: largest demand first, cheapest feasible facility."""
    n = len(demands)
    result = np.zeros(n, dtype=int)
    load = np.zeros(len(capacities))
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    return result


def _total_cost(demands, capacities, open_cost, assign_cost, result):
    """Compute total cost for a given assignment."""
    load = np.zeros(len(capacities))
    for i in range(len(demands)):
        load[result[i]] += demands[i]
    # Check validity
    for f in range(len(capacities)):
        if load[f] > capacities[f]:
            return float('inf')
    open_mask = load > 0
    cost = np.sum(open_cost[open_mask])
    cost += np.sum(assign_cost[np.arange(len(demands)), result])
    return cost


def _local_search(demands, capacities, open_cost, assign_cost, result, time_limit=80.0):
    """Local search: try moving each customer to a different facility if it improves cost."""
    n = len(demands)
    m = len(capacities)
    start_time = time.time()
    result = result.copy()
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    
    best_cost = _total_cost(demands, capacities, open_cost, assign_cost, result)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try moving each customer
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            current_f = result[i]
            current_cost = assign_cost[i, current_f]
            
            # Try all other facilities
            for f in range(m):
                if f == current_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                new_cost = assign_cost[i, f]
                # Check if moving opens/closes a facility
                old_open = load[current_f] - demands[i] > 0
                new_open = load[f] + demands[i] > 0
                
                delta = new_cost - current_cost
                if not old_open and load[current_f] - demands[i] == 0:
                    delta -= open_cost[current_f]
                if not new_open and load[f] == 0:
                    delta += open_cost[f]
                
                if delta < -1e-9:
                    # Make the move
                    load[current_f] -= demands[i]
                    load[f] += demands[i]
                    result[i] = f
                    best_cost += delta
                    improved = True
                    break
    
    return result, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    """Assign each customer to a facility, minimizing total cost."""
    start_time = time.time()
    n = len(demands)
    m = len(capacities)
    
    # Initial greedy solution
    result = _greedy_assign(demands, capacities, open_cost, assign_cost)
    
    # Verify validity
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    for f in range(m):
        if load[f] > capacities[f]:
            # Fallback: assign to any facility with capacity
            load = np.zeros(m)
            for i in range(n):
                for f in range(m):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        break
            break
    
    # Local search refinement
    result, _ = _local_search(demands, capacities, open_cost, assign_cost, result, time_limit=80.0)
    
    # Final validity check
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    for f in range(m):
        if load[f] > capacities[f]:
            # Should not happen, but fallback
            return _greedy_assign(demands, capacities, open_cost, assign_cost)
    
    return result
# EVOLVE-BLOCK-END
