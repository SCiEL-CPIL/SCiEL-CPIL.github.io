import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Greedy Construction
    # Sort customers by demand descending to handle large demands first
    order = np.argsort(-demands)
    
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # Precompute min costs to identify "tight" customers? 
    # For now, just process in order of demand.
    
    for i in order:
        min_eff_cost = float('inf')
        best_f = -1
        
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                # Amortized open cost: if facility is empty, the open cost is "paid" by this customer (and future ones).
                # Using (1 + load) as a denominator approximates the average cost per customer if load stays constant.
                # If load is 0, cost is open_cost. If load is large, cost is small.
                # This encourages opening a facility if the assignment cost is low enough to offset the open cost share.
                amortized_open = open_cost[f] / (1 + load[f])
                eff_cost = assign_cost[i, f] + amortized_open
                
                if eff_cost < min_eff_cost:
                    min_eff_cost = eff_cost
                    best_f = f
        
        if best_f == -1:
            # Fallback: find any feasible facility
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    best_f = f
                    break
            if best_f == -1:
                # Should not happen if total demand <= total capacity
                best_f = 0
                
        result[i] = best_f
        load[best_f] += demands[i]
        
    # 2. Local Search (Deterministic Descent)
    # We perform 1-move and 1-swap improvements until no more improvement is found.
    
    def compute_total_cost(assignment, load):
        cost = 0.0
        for f in range(n_facilities):
            if load[f] > 0:
                cost += open_cost[f]
        for i in range(n_customers):
            cost += assign_cost[i, assignment[i]]
        return cost

    current_assignment = result.copy()
    current_load = load.copy()
    current_cost = compute_total_cost(current_assignment, current_load)
    
    improved = True
    max_passes = 50 # Safety limit
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        # 1-Move Search
        for i in range(n_customers):
            old_f = current_assignment[i]
            best_delta = 0.0
            best_new_f = old_f
            
            for new_f in range(n_facilities):
                if new_f == old_f:
                    continue
                if current_load[new_f] + demands[i] > capacities[new_f]:
                    continue
                
                # Calculate delta
                delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                delta_open = 0.0
                if current_load[old_f] == demands[i]:
                    delta_open -= open_cost[old_f]
                if current_load[new_f] == 0:
                    delta_open += open_cost[new_f]
                
                delta = delta_assign + delta_open
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_f = new_f
            
            if best_new_f != old_f:
                # Apply move
                current_load[old_f] -= demands[i]
                current_load[best_new_f] += demands[i]
                current_assignment[i] = best_new_f
                current_cost += best_delta
                improved = True
        
        # 1-Swap Search
        # To make this efficient, we can just iterate over pairs. 100 customers -> 4950 pairs.
        # 20 facilities. This is fast.
        for i in range(n_customers):
            f_i = current_assignment[i]
            for k in range(i + 1, n_customers):
                f_k = current_assignment[k]
                if f_i == f_k:
                    continue
                
                # Check capacity after swap
                # f_k gets i, loses k
                if current_load[f_k] - demands[k] + demands[i] > capacities[f_k]:
                    continue
                # f_i gets k, loses i
                if current_load[f_i] - demands[i] + demands[k] > capacities[f_i]:
                    continue
                
                # Calculate delta
                delta_assign = (assign_cost[i, f_k] + assign_cost[k, f_i]) - (assign_cost[i, f_i] + assign_cost[k, f_k])
                
                # Open cost change for swap:
                # f_i loses i, gains k. 
                # If load[f_i] == demands[i], it was only i. Now it has k. Still open (unless demands[k]==0, but demands>=5).
                # If load[f_k] == demands[k], it was only k. Now it has i. Still open.
                # So open cost does NOT change in a swap between two distinct facilities where both customers move.
                # Wait, what if one facility becomes empty?
                # If load[f_i] == demands[i], then f_i has only i. After swap, f_i has k. Load becomes demands[k] > 0. So it stays open.
                # If load[f_k] == demands[k], then f_k has only k. After swap, f_k has i. Load becomes demands[i] > 0. So it stays open.
                # So delta_open is always 0 for valid swaps.
                
                delta = delta_assign
                
                if delta < 0:
                    # Apply swap
                    current_assignment[i] = f_k
                    current_assignment[k] = f_i
                    current_load[f_i] = current_load[f_i] - demands[i] + demands[k]
                    current_load[f_k] = current_load[f_k] - demands[k] + demands[i]
                    current_cost += delta
                    improved = True
                    # Note: After swap, indices i and k are swapped. 
                    # The loop continues with k+1. The new assignment for i is f_k, for k is f_i.
                    # We don't need to re-evaluate (i, k) immediately.
                    
    return current_assignment
