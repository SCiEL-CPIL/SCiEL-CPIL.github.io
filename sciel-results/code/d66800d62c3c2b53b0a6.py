"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Phase 1: Greedy assignment with open-cost-aware scoring
    # Sort customers by demand descending
    order = np.argsort(-demands)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=float)
    count = np.zeros(n_facilities, dtype=int)  # number of customers assigned to each facility
    
    for i in order:
        best_f = -1
        best_score = np.inf
        for f in range(n_facilities):
            if load[f] + demands[i] > capacities[f]:
                continue
            # Score: assignment cost + prorated open cost
            # First customer bears full open cost, subsequent bear open_cost / (count+1)
            if count[f] == 0:
                score = assign_cost[i, f] + open_cost[f]
            else:
                score = assign_cost[i, f] + open_cost[f] / (count[f] + 1)
            if score < best_score:
                best_score = score
                best_f = f
        if best_f == -1:
            # Should not happen if capacities are sufficient, but fallback
            # Try any feasible facility
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    best_f = f
                    break
            if best_f == -1:
                # No feasible facility - assign to facility with most remaining capacity
                best_f = np.argmax(capacities - load)
        result[i] = best_f
        load[best_f] += demands[i]
        count[best_f] += 1
    
    # Phase 2: Local search improvement
    # Compute initial total cost
    open_mask = (count > 0)
    total_cost = np.sum(open_cost[open_mask]) + np.sum(assign_cost[np.arange(n_customers), result])
    
    max_iterations = 50
    for iteration in range(max_iterations):
        improved = False
        # Try moving each customer to a better facility
        for i in range(n_customers):
            old_f = result[i]
            best_new_f = -1
            best_new_cost = total_cost
            
            for f in range(n_facilities):
                if f == old_f:
                    continue
                # Check capacity
                if load[f] + demands[i] > capacities[f]:
                    continue
                
                # Compute cost change
                # Assignment cost change
                assign_change = assign_cost[i, f] - assign_cost[i, old_f]
                
                # Open cost change
                open_change = 0.0
                # If old facility becomes empty, we save its open cost
                if count[old_f] == 1:
                    open_change -= open_cost[old_f]
                # If new facility was empty, we add its open cost
                if count[f] == 0:
                    open_change += open_cost[f]
                
                total_change = assign_change + open_change
                new_total = total_cost + total_change
                
                if new_total < best_new_cost - 1e-9:
                    best_new_cost = new_total
                    best_new_f = f
            
            if best_new_f != -1:
                # Perform the move
                result[i] = best_new_f
                load[old_f] -= demands[i]
                load[best_new_f] += demands[i]
                count[old_f] -= 1
                count[best_new_f] += 1
                total_cost = best_new_cost
                improved = True
        
        if not improved:
            break
    
    return result
# EVOLVE-BLOCK-END
