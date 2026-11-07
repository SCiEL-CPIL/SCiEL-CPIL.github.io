"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Deadline guard: 150s wall-clock, use 140s for safety
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial greedy assignment: largest demand first, cheapest feasible facility
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    
    # Local search: try to improve by moving or swapping customers
    def calc_cost(assignment):
        total_assign = np.sum(assign_cost[np.arange(n_customers), assignment])
        used = set(assignment.tolist())
        total_open = np.sum(open_cost[list(used)])
        return total_assign + total_open
    
    def is_valid(assignment):
        for f in range(n_facilities):
            mask = assignment == f
            if mask.any():
                if demands[mask].sum() > capacities[f]:
                    return False
        return True
    
    best_assignment = result.copy()
    best_cost = calc_cost(best_assignment)
    
    # Local search with moves and swaps
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(n_customers):
            if time.time() > deadline - 10.0:
                break
            current_f = best_assignment[i]
            # Try moving customer i to another facility
            for f in range(n_facilities):
                if f == current_f:
                    continue
                # Check capacity constraint
                if load[f] + demands[i] <= capacities[f]:
                    new_cost = best_cost - assign_cost[i, current_f] + assign_cost[i, f]
                    # Account for open cost changes
                    if current_f != f:
                        # Check if current_f becomes empty
                        others_at_current = np.sum(best_assignment == current_f)
                        others_at_new = np.sum(best_assignment == f)
                        if others_at_current == 1:
                            new_cost -= open_cost[current_f]
                        if others_at_new == 0:
                            new_cost += open_cost[f]
                    if new_cost < best_cost - 1e-9:
                        # Make the move
                        load[current_f] -= demands[i]
                        load[f] += demands[i]
                        best_assignment[i] = f
                        best_cost = new_cost
                        improved = True
                        break
        
        # Try swaps between pairs of customers
        if time.time() > deadline - 10.0:
            break
        for i in range(n_customers):
            if time.time() > deadline - 10.0:
                break
            for j in range(i + 1, n_customers):
                if time.time() > deadline - 10.0:
                    break
                fi = best_assignment[i]
                fj = best_assignment[j]
                if fi == fj:
                    continue
                # Check capacity constraints for swap
                if load[fi] - demands[i] + demands[j] <= capacities[fi] and \
                   load[fj] - demands[j] + demands[i] <= capacities[fj]:
                    new_cost = best_cost - assign_cost[i, fi] - assign_cost[j, fj] + \
                              assign_cost[i, fj] + assign_cost[j, fi]
                    # Open cost doesn't change in a swap (both facilities still used)
                    if new_cost < best_cost - 1e-9:
                        # Make the swap
                        load[fi] = load[fi] - demands[i] + demands[j]
                        load[fj] = load[fj] - demands[j] + demands[i]
                        best_assignment[i] = fj
                        best_assignment[j] = fi
                        best_cost = new_cost
                        improved = True
    
    # Final verification
    if not is_valid(best_assignment):
        # Fallback to greedy result
        return result
    
    return best_assignment
# EVOLVE-BLOCK-END
