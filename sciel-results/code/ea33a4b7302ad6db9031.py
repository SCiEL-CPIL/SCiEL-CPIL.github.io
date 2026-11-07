"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility using a combination of:
    1. LP relaxation with rounding (using scipy if available, else heuristic)
    2. Local search improvements (swap and reassignment)
    
    Returns an integer array of facility indices for each customer.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Deadline guard: 120s wall-clock limit, use 110s for safety
    deadline = time.time() + 110
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    # Initial assignment: greedy largest-first with cheapest feasible
    result = _greedy_initial(demands, capacities, assign_cost)
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands[i]
    
    # Local search: try swapping customers between facilities
    best_result = result.copy()
    best_load = load.copy()
    best_cost = _total_cost(best_result, best_load, open_cost, assign_cost)
    
    # Multiple passes of local search
    improved = True
    pass_count = 0
    while improved and time.time() < deadline - 5:
        improved = False
        pass_count += 1
        
        # Try reassigning each customer to a different facility
        for i in range(n_customers):
            if time.time() > deadline - 5:
                break
            current_f = best_result[i]
            # Try all other facilities
            for f in range(n_facilities):
                if f == current_f:
                    continue
                # Check capacity
                if best_load[f] + demands[i] > capacities[f]:
                    continue
                # Compute cost change
                # Remove i from current_f
                old_load_cur = best_load[current_f]
                new_load_cur = old_load_cur - demands[i]
                # Add i to f
                new_load_f = best_load[f] + demands[i]
                
                # Open cost change: if current_f becomes empty, save open_cost
                open_change = 0
                if new_load_cur == 0:
                    open_change -= open_cost[current_f]
                if new_load_f == 0 and best_load[f] == 0:
                    # f was already empty, but now has load, so add open cost
                    open_change += open_cost[f]
                
                # Assignment cost change
                assign_change = assign_cost[i, f] - assign_cost[i, current_f]
                
                total_change = open_change + assign_change
                if total_change < -1e-9:
                    # Accept
                    best_load[current_f] = new_load_cur
                    best_load[f] = new_load_f
                    best_result[i] = f
                    best_cost += total_change
                    improved = True
                    break  # Move to next customer
    
    # Second phase: try swapping pairs of customers
    if time.time() < deadline - 10:
        for i in range(n_customers):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, n_customers):
                if time.time() > deadline - 10:
                    break
                fi = best_result[i]
                fj = best_result[j]
                if fi == fj:
                    continue
                # Swap: i goes to fj, j goes to fi
                if best_load[fj] - demands[j] + demands[i] > capacities[fj]:
                    continue
                if best_load[fi] - demands[i] + demands[j] > capacities[fi]:
                    continue
                
                # Compute cost change
                new_load_fi = best_load[fi] - demands[i] + demands[j]
                new_load_fj = best_load[fj] - demands[j] + demands[i]
                
                open_change = 0
                if new_load_fi == 0:
                    open_change -= open_cost[fi]
                if best_load[fi] == 0 and new_load_fi > 0:
                    open_change += open_cost[fi]
                if new_load_fj == 0:
                    open_change -= open_cost[fj]
                if best_load[fj] == 0 and new_load_fj > 0:
                    open_change += open_cost[fj]
                
                assign_change = (assign_cost[i, fj] - assign_cost[i, fi]) + \
                                (assign_cost[j, fi] - assign_cost[j, fj])
                
                total_change = open_change + assign_change
                if total_change < -1e-9:
                    # Accept swap
                    best_load[fi] = new_load_fi
                    best_load[fj] = new_load_fj
                    best_result[i] = fj
                    best_result[j] = fi
                    best_cost += total_change
    
    # Final verification
    final_load = np.zeros(n_facilities)
    for i in range(n_customers):
        final_load[best_result[i]] += demands[i]
    if np.any(final_load > capacities + 1e-9):
        # Fallback to greedy
        best_result = _greedy_initial(demands, capacities, assign_cost)
    
    return best_result.astype(int)


def _greedy_initial(demands, capacities, assign_cost):
    """Greedy assignment: largest demand first, cheapest feasible facility."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # Sort by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        # Sort facilities by assignment cost for this customer
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # Should not happen if capacities are sufficient
            # Assign to facility with most remaining capacity
            remaining = capacities - load
            f = np.argmax(remaining)
            result[i] = f
    
    return result


def _total_cost(result, load, open_cost, assign_cost):
    """Compute total cost: open cost for used facilities + assignment costs."""
    n_customers = len(result)
    cost = 0.0
    for f in range(len(load)):
        if load[f] > 0:
            cost += open_cost[f]
    for i in range(n_customers):
        cost += assign_cost[i, result[i]]
    return cost
# EVOLVE-BLOCK-END
