# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities with capacity constraints, minimizing total cost.
    
    Strategy:
    1. Greedy seed: sort by demand descending, assign to cheapest feasible facility.
    2. Local search: try moving/reassigning customers to reduce cost while respecting capacities.
    3. Deadline guard: stop if approaching 120s limit.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Greedy seed
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=float)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        # Get facilities sorted by assignment cost for this customer
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                assignment[i] = f
                break
        else:
            # Should not happen if problem is feasible, but fallback
            assignment[i] = 0
            load[0] += demands[i]
    
    def total_cost(assignment):
        """Compute total cost for a given assignment."""
        # Assignment costs
        assign_total = 0.0
        for i in range(n_customers):
            assign_total += assign_cost[i, assignment[i]]
        
        # Open costs
        open_total = 0.0
        used = set(assignment.tolist())
        for f in used:
            open_total += open_cost[f]
        
        return assign_total + open_total
    
    best_cost = total_cost(assignment)
    best_assignment = assignment.copy()
    
    # Local search: try moving each customer to a different facility
    # and also try swapping pairs of customers between facilities
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try single moves
        for i in range(n_customers):
            if time.time() > deadline:
                break
            
            current_f = assignment[i]
            # Try all other facilities
            for f in range(n_facilities):
                if f == current_f:
                    continue
                
                # Check capacity
                if load[f] + demands[i] > capacities[f]:
                    continue
                
                # Compute cost delta
                # Remove i from current_f, add to f
                # Assignment cost change
                delta_assign = assign_cost[i, f] - assign_cost[i, current_f]
                
                # Open cost change: 
                # - If current_f becomes empty after removing i, we save open_cost[current_f]
                # - If f was empty before adding i, we add open_cost[f]
                count_current = np.sum(assignment == current_f)
                count_f = np.sum(assignment == f)
                
                delta_open = 0.0
                if count_current == 1:  # current_f will become empty
                    delta_open -= open_cost[current_f]
                if count_f == 0:  # f was empty
                    delta_open += open_cost[f]
                
                delta_total = delta_assign + delta_open
                
                if delta_total < -1e-9:
                    # Apply the move
                    load[current_f] -= demands[i]
                    load[f] += demands[i]
                    assignment[i] = f
                    best_cost += delta_total
                    best_assignment = assignment.copy()
                    improved = True
                    break  # Restart iteration after a move
        
        # Try pair swaps: swap two customers between two facilities
        if time.time() > deadline:
            break
            
        for i in range(n_customers):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_customers):
                if time.time() > deadline:
                    break
                
                fi = assignment[i]
                fj = assignment[j]
                
                if fi == fj:
                    continue
                
                # Check if swap is feasible
                # i goes to fj, j goes to fi
                # Need: load[fj] - demands[j] + demands[i] <= capacities[fj]
                # Need: load[fi] - demands[i] + demands[j] <= capacities[fi]
                
                if load[fj] - demands[j] + demands[i] > capacities[fj]:
                    continue
                if load[fi] - demands[i] + demands[j] > capacities[fi]:
                    continue
                
                # Compute cost delta
                delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                               (assign_cost[i, fi] + assign_cost[j, fj])
                
                # Open cost change
                # fi loses i, gains j; fj loses j, gains i
                # fi becomes empty only if count[fi] == 1 (but we're keeping fi non-empty since j moves in)
                # Actually, fi count changes from count[fi] to count[fi] - 1 + 1 = count[fi] (no change)
                # Same for fj. So open cost doesn't change for swaps between non-empty facilities.
                # But if one facility becomes empty... wait, in a swap, both facilities keep at least one customer
                # unless one of them had exactly one customer. Let's handle that.
                
                count_fi = np.sum(assignment == fi)
                count_fj = np.sum(assignment == fj)
                
                delta_open = 0.0
                # fi: loses i, gains j. If count_fi == 1, then after removing i it's empty, but j comes in, so still non-empty
                # So fi never becomes empty in a swap. Same for fj.
                # Open cost doesn't change for swaps.
                
                delta_total = delta_assign + delta_open
                
                if delta_total < -1e-9:
                    # Apply swap
                    load[fi] = load[fi] - demands[i] + demands[j]
                    load[fj] = load[fj] - demands[j] + demands[i]
                    assignment[i] = fj
                    assignment[j] = fi
                    best_cost += delta_total
                    best_assignment = assignment.copy()
                    improved = True
                    break
            else:
                continue
            break
    
    return best_assignment
# EVOLVE-BLOCK-END
