"""Capacitated facility location: vectorized heuristic + local search with time budget."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _true_obj(result, demands, capacities, open_cost, assign_cost):
    """Compute objective: open costs + assignment costs. Returns inf if infeasible."""
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    if np.any(load > capacities):
        return float('inf')
    open_sum = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            open_sum += open_cost[f]
    assign_sum = float(np.sum(assign_cost[np.arange(len(result)), result]))
    return open_sum + assign_sum


def _vectorized_seed(demands, capacities, assign_cost):
    """Vectorized heuristic: assign all customers to their cheapest facility, then repair capacity violations."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Start with each customer assigned to their cheapest facility
    result = np.argmin(assign_cost, axis=1)
    load = np.zeros(n_fac)
    for c in range(n_cust):
        load[result[c]] += demands[c]
    
    # Repair: move customers from overloaded facilities to others
    # Process facilities in order of most overloaded first
    for _ in range(n_fac * 3):  # Max iterations for repair
        # Find facilities with excess load
        excess_mask = load > capacities
        if not np.any(excess_mask):
            break
        
        # For each overloaded facility, move customers with highest demand to other facilities
        for f in np.where(excess_mask)[0]:
            while load[f] > capacities[f]:
                # Find customers at facility f
                customers_at_f = np.where(result == f)[0]
                if len(customers_at_f) == 0:
                    break
                
                # Pick the customer with highest demand (easiest to move)
                c = customers_at_f[np.argmax(demands[customers_at_f])]
                d = demands[c]
                
                # Find feasible alternative facilities sorted by cost
                # Exclude current facility
                alt_costs = assign_cost[c].copy()
                alt_costs[f] = np.inf
                alt_order = np.argsort(alt_costs)
                
                moved = False
                for af in alt_order:
                    if load[af] + d <= capacities[af]:
                        # Move customer
                        load[f] -= d
                        load[af] += d
                        result[c] = af
                        moved = True
                        break
                
                if not moved:
                    # Cannot fix this violation; break to avoid infinite loop
                    break
    
    # Final feasibility check - if still infeasible, use greedy as fallback
    if np.any(load > capacities):
        return _greedy_seed(demands, capacities, assign_cost)
    
    return result


def _greedy_seed(demands, capacities, assign_cost):
    """Largest-first greedy assignment (fallback)."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        d = demands[i]
        feasible = []
        for f in np.argsort(assign_cost[i]):
            if load[f] + d <= capacities[f]:
                feasible.append(f)
        if feasible:
            result[i] = feasible[0]
            load[result[i]] += d
        else:
            result[i] = 0
            load[0] += d
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, deadline):
    """Local search: try moving customers to other facilities or swapping pairs."""
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    
    best_obj = _true_obj(result, demands, capacities, open_cost, assign_cost)
    if best_obj == float('inf'):
        return result, best_obj
    
    # Precompute current assignment costs per customer
    curr_assign_cost = assign_cost[np.arange(n_cust), result].copy()
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        
        # Try single moves: move customer c from result[c] to facility f
        for c in range(n_cust):
            if time.time() >= deadline:
                break
            d = demands[c]
            cur_f = result[c]
            cur_cost_c = assign_cost[c, cur_f]
            
            # Try all other facilities
            for f in range(n_fac):
                if f == cur_f:
                    continue
                # Check capacity: new facility has room
                if load[f] + d > capacities[f]:
                    continue
                
                # Compute delta in objective
                # Assignment cost change
                delta_assign = assign_cost[c, f] - cur_cost_c
                
                # Open cost change:
                # If cur_f loses its last customer, we save open_cost[cur_f]
                # If f gains its first customer, we pay open_cost[f]
                delta_open = 0.0
                if load[cur_f] == d:  # cur_f becomes empty
                    delta_open -= open_cost[cur_f]
                if load[f] == 0:  # f becomes non-empty
                    delta_open += open_cost[f]
                
                delta = delta_assign + delta_open
                
                if delta < -1e-9:
                    # Apply move
                    result[c] = f
                    load[cur_f] -= d
                    load[f] += d
                    curr_assign_cost[c] = assign_cost[c, f]
                    best_obj += delta
                    improved = True
                    no_improve = 0
                    break  # Move this customer to next customer
        
        if not improved:
            no_improve += 1
        
        # Try pair swaps: swap customers c1 and c2
        if no_improve < max_no_improve // 2:
            for c1 in range(n_cust):
                if time.time() >= deadline:
                    break
                for c2 in range(c1 + 1, n_cust):
                    if time.time() >= deadline:
                        break
                    f1 = result[c1]
                    f2 = result[c2]
                    if f1 == f2:
                        continue
                    d1 = demands[c1]
                    d2 = demands[c2]
                    
                    # After swap: c1 goes to f2, c2 goes to f1
                    # Check capacities
                    # f1 loses d1, gains d2: new load = load[f1] - d1 + d2
                    # f2 loses d2, gains d1: new load = load[f2] - d2 + d1
                    new_load_f1 = load[f1] - d1 + d2
                    new_load_f2 = load[f2] - d2 + d1
                    if new_load_f1 > capacities[f1] or new_load_f2 > capacities[f2]:
                        continue
                    
                    # Compute delta
                    delta_assign = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                                   (assign_cost[c1, f1] + assign_cost[c2, f2])
                    
                    # Open cost changes
                    delta_open = 0.0
                    # f1: was load[f1], now new_load_f1
                    if load[f1] == d1:  # f1 becomes empty after removing c1
                        if new_load_f1 > 0:  # but c2 comes in
                            # f1 stays open, no change
                            pass
                        else:
                            delta_open -= open_cost[f1]
                    # f2: was load[f2], now new_load_f2
                    if load[f2] == d2:  # f2 becomes empty after removing c2
                        if new_load_f2 > 0:  # but c1 comes in
                            pass
                        else:
                            delta_open -= open_cost[f2]
                    
                    # More careful: track which facilities change open status
                    # f1: old load = load[f1], new load = new_load_f1
                    # f2: old load = load[f2], new load = new_load_f2
                    # If old load > 0 and new load == 0: save open cost
                    # If old load == 0 and new load > 0: pay open cost
                    # But in swap, f1 loses d1 and gains d2, f2 loses d2 and gains d1
                    # So f1's load changes from L1 to L1 - d1 + d2
                    # f2's load changes from L2 to L2 - d2 + d1
                    
                    delta_open = 0.0
                    if load[f1] > 0 and new_load_f1 == 0:
                        delta_open -= open_cost[f1]
                    elif load[f1] == 0 and new_load_f1 > 0:
                        delta_open += open_cost[f1]
                    if load[f2] > 0 and new_load_f2 == 0:
                        delta_open -= open_cost[f2]
                    elif load[f2] == 0 and new_load_f2 > 0:
                        delta_open += open_cost[f2]
                    
                    delta = delta_assign + delta_open
                    
                    if delta < -1e-9:
                        # Apply swap
                        result[c1], result[c2] = f2, f1
                        load[f1] = new_load_f1
                        load[f2] = new_load_f2
                        curr_assign_cost[c1] = assign_cost[c1, f2]
                        curr_assign_cost[c2] = assign_cost[c2, f1]
                        best_obj += delta
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        if not improved:
            no_improve += 1
    
    return result, best_obj


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using vectorized seed + local search.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Safety margin: 150s limit, aim for 145s max
    deadline = time.time() + 145.0
    
    # Vectorized seed (cheapest facility + capacity repair)
    result = _vectorized_seed(demands, capacities, assign_cost)
    
    # Local search with time budget
    result, best_obj = _local_search(result, demands, capacities, open_cost, assign_cost, deadline)
    
    # Verify feasibility
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    if np.any(load > capacities):
        # Fallback: return greedy result
        return _greedy_seed(demands, capacities, assign_cost)
    
    return result.astype(int)
# EVOLVE-BLOCK-END
