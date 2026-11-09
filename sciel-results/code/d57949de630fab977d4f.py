import numpy as np
import random
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Initial Solution: Best-Fit Decreasing Heuristic
    # Sort customers by demand descending
    order = np.argsort(-demands)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in order:
        best_f = -1
        best_remaining = -1
        # Find facility with most remaining capacity that fits, 
        # preferring lower assign_cost among those? 
        # Let's just pick the one with lowest assign_cost among feasible, 
        # tie-break by remaining capacity (largest first to balance load?) 
        # Or just lowest cost.
        min_cost = float('inf')
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                if assign_cost[i, f] < min_cost:
                    min_cost = assign_cost[i, f]
                    best_f = f
        if best_f == -1:
            # Fallback: find any feasible
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    best_f = f
                    break
            if best_f == -1:
                # Should not happen if total demand <= total capacity
                # Force assign to 0 (invalid but better than crash, evaluator will mark invalid)
                best_f = 0
        result[i] = best_f
        load[best_f] += demands[i]
        
    # Calculate initial cost
    def get_cost(assignment, load):
        cost = 0.0
        for f in range(n_facilities):
            if load[f] > 0:
                cost += open_cost[f]
        for i in range(n_cust):
            cost += assign_cost[i, assignment[i]]
        return cost

    # Wait, the inner function needs n_customers. Let's define it outside or pass it.
    def compute_cost(assignment, load):
        cost = 0.0
        for f in range(n_facilities):
            if load[f] > 0:
                cost += open_cost[f]
        for i in range(n_customers):
            cost += assign_cost[i, assignment[i]]
        return cost

    current_cost = compute_cost(result, load)
    
    # 2. Simulated Annealing
    T = 100.0 # Initial temperature
    T_min = 1e-4
    cooling_rate = 0.999
    max_iter = 50000 # Adjust based on time
    start_time = time.time()
    time_limit = 100.0 # Leave 20s buffer
    
    current_assignment = result.copy()
    current_load = load.copy()
    best_assignment = result.copy()
    best_cost = current_cost
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Generate a neighbor
        move_type = random.choice(['move', 'swap'])
        
        if move_type == 'move':
            i = random.randint(0, n_customers - 1)
            old_f = current_assignment[i]
            # Pick a random other facility
            new_f = random.randint(0, n_facilities - 1)
            while new_f == old_f:
                new_f = random.randint(0, n_facilities - 1)
            
            # Check capacity
            if current_load[new_f] + demands[i] > capacities[new_f]:
                continue
            
            # Calculate Delta
            # Cost change in assignment
            delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
            
            # Cost change in open cost
            delta_open = 0.0
            # Old facility becomes empty?
            if current_load[old_f] == demands[i]:
                delta_open -= open_cost[old_f]
            # New facility was empty?
            if current_load[new_f] == 0:
                delta_open += open_cost[new_f]
            
            delta = delta_assign + delta_open
            
            # Acceptance
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Apply
                current_assignment[i] = new_f
                current_load[old_f] -= demands[i]
                current_load[new_f] += demands[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = current_assignment.copy()
                    
        else: # swap
            i = random.randint(0, n_customers - 1)
            k = random.randint(0, n_customers - 1)
            if i == k:
                continue
            f_i = current_assignment[i]
            f_k = current_assignment[k]
            if f_i == f_k:
                continue
            
            # Check capacity after swap
            # f_k gets i, loses k. Load change: +demands[i] - demands[k]
            # f_i gets k, loses i. Load change: +demands[k] - demands[i]
            if current_load[f_k] - demands[k] + demands[i] > capacities[f_k]:
                continue
            if current_load[f_i] - demands[i] + demands[k] > capacities[f_i]:
                continue
            
            # Calculate Delta
            delta_assign = (assign_cost[i, f_k] + assign_cost[k, f_i]) - (assign_cost[i, f_i] + assign_cost[k, f_k])
            
            # Open cost:
            # Do any facilities become empty?
            # f_i loses i, gains k. If load[f_i] was demands[i] (only i), and demands[k] > 0, it's not empty.
            # If load[f_i] was demands[i] + others, it's not empty.
            # It only becomes empty if load[f_i] == demands[i] AND we move i out. But we swap, so k comes in.
            # So f_i is never empty after swap if it wasn't empty before?
            # Wait. If load[f_i] == demands[i], then after swap, load[f_i] = demands[k]. Since demands[k] > 0, it's not empty.
            # So open costs do NOT change in a swap between two occupied facilities.
            # Exception: If one facility had ONLY that customer?
            # If load[f_i] == demands[i], then f_i has only i. After swap, f_i has k. Still open.
            # If load[f_k] == demands[k], then f_k has only k. After swap, f_k has i. Still open.
            # So delta_open is always 0 for swaps between distinct facilities where both had at least that customer.
            # Wait, what if f_i == f_k? We checked that.
            # So delta_open = 0.
            
            delta = delta_assign
            
            # Acceptance
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Apply
                current_assignment[i] = f_k
                current_assignment[k] = f_i
                current_load[f_i] = current_load[f_i] - demands[i] + demands[k]
                current_load[f_k] = current_load[f_k] - demands[k] + demands[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = current_assignment.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_assignment
