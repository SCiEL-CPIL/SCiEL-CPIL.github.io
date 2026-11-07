"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using LP relaxation of CFLP
    followed by greedy assignment and local search.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Formulate and solve LP relaxation
    # Variables: x_{ij} (n_customers * n_facilities), y_j (n_facilities)
    n_x = n_customers * n_facilities
    n_y = n_facilities
    n_vars = n_x + n_y
    
    # Objective coefficients
    c = np.zeros(n_vars)
    c[:n_x] = assign_cost.flatten()
    c[n_x:] = open_cost
    
    # Constraints
    # 1. Each customer assigned to exactly one facility: sum_j x_{ij} = 1
    A_eq = np.zeros((n_customers, n_vars))
    for i in range(n_customers):
        for j in range(n_facilities):
            idx = i * n_facilities + j
            A_eq[i, idx] = 1.0
            
    # 2. Capacity: sum_i d_i x_{ij} - C_j y_j <= 0
    A_ub = np.zeros((n_facilities, n_vars))
    for j in range(n_facilities):
        for i in range(n_customers):
            idx = i * n_facilities + j
            A_ub[j, idx] = demands[i]
        A_ub[j, n_x + j] = -capacities[j]
        
    # Bounds
    bounds = scipy.optimize.Bounds(0, 1)
    
    # Solve LP
    try:
        res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=np.zeros(n_facilities), 
                                     A_eq=A_eq, b_eq=np.ones(n_customers), 
                                     bounds=bounds, method='highs')
        
        if res.status != 0:
            # Fallback if LP fails
            return _fallback_heuristic(demands, capacities, open_cost, assign_cost)
            
        x_flat = res.x[:n_x]
        y_vals = res.x[n_x:]
        
        # 2. Post-processing: Determine open facilities
        # Threshold for opening
        open_threshold = 0.5
        open_facilities = np.where(y_vals > open_threshold)[0]
        
        # If no facilities opened (should not happen if feasible), open all
        if len(open_facilities) == 0:
            open_facilities = np.arange(n_facilities)
            
        # 3. Greedy Assignment to open facilities
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        
        # Sort customers by demand descending to handle large items first?
        # Or sort by "regret" or LP value? 
        # Let's try sorting by demand descending first, as it's a standard packing heuristic.
        order = np.argsort(-demands)
        
        # We want to assign to open facilities.
        # We can use the LP solution x_{ij} to guide the choice.
        # For each customer, among open facilities with capacity, pick the one with highest x_{ij} or lowest cost?
        # Lowest cost is safer for optimality if LP is good.
        
        for i in order:
            best_fac = -1
            best_score = np.inf
            
            # Only consider open facilities
            for j in open_facilities:
                if load[j] + demands[i] <= capacities[j]:
                    # Score: assign_cost[i, j]
                    # We can also incorporate LP value to break ties or guide if costs are similar
                    score = assign_cost[i, j]
                    if score < best_score:
                        best_score = score
                        best_fac = j
            
            if best_fac == -1:
                # If no open facility has capacity, we need to open a new one or reassign.
                # This indicates the LP solution's set of open facilities was insufficient for this greedy order.
                # Strategy: Open the facility that minimizes cost + open_cost for this customer, 
                # provided it has capacity.
                # Or, more robustly, find any facility (open or closed) that can take it.
                # If we open a new one, we pay open_cost.
                
                # Let's try to find the best facility among ALL facilities that has capacity.
                # If it's closed, we "open" it in our heuristic (add to open_facilities).
                all_fac = np.arange(n_facilities)
                for j in all_fac:
                    if load[j] + demands[i] <= capacities[j]:
                        # Cost is assign_cost + open_cost if not already open
                        is_open = (j in open_facilities)
                        cost = assign_cost[i, j]
                        if not is_open:
                            cost += open_cost[j]
                        
                        if cost < best_score:
                            best_score = cost
                            best_fac = j
                
                if best_fac == -1:
                    # Infeasible instance? Or greedy failed completely.
                    # Fallback to robust heuristic
                    return _fallback_heuristic(demands, capacities, open_cost, assign_cost)
                
                # If we picked a closed facility, add it to open list
                if best_fac not in open_facilities:
                    open_facilities = np.append(open_facilities, best_fac)
                    
            result[i] = best_fac
            load[best_fac] += demands[i]
            
        # 4. Local Search to improve cost
        result = _local_search(demands, capacities, open_cost, assign_cost, result)
        
        return result
        
    except Exception:
        return _fallback_heuristic(demands, capacities, open_cost, assign_cost)


def _fallback_heuristic(demands, capacities, open_cost, assign_cost):
    """
    Robust greedy heuristic with local search.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        best_fac = -1
        best_cost = np.inf
        
        for j in range(n_facilities):
            if load[j] + demands[i] <= capacities[j]:
                # Effective cost: assignment cost + open cost if facility is new
                # We don't track "new" strictly here, but we can approximate
                # by checking if load[j] was 0 before this assignment.
                # However, for the first pass, let's just use assign_cost.
                # To be smarter, we should account for open cost.
                # If load[j] == 0, we pay open_cost[j].
                cost = assign_cost[i, j]
                if load[j] == 0:
                    cost += open_cost[j]
                    
                if cost < best_cost:
                    best_cost = cost
                    best_fac = j
                    
        if best_fac == -1:
            # No feasible facility. This should not happen for valid instances.
            # Pick facility with most remaining capacity as last resort.
            rem_cap = capacities - load
            best_fac = np.argmax(rem_cap)
            
        result[i] = best_fac
        load[best_fac] += demands[i]
        
    # Local search
    result = _local_search(demands, capacities, open_cost, assign_cost, result)
    
    return result


def _local_search(demands, capacities, open_cost, assign_cost, result):
    """
    Local search to improve the assignment.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands[i]
        
    # Helper to calculate total cost
    def calc_cost(assignments):
        total_assign = np.sum(assign_cost[np.arange(n_customers), assignments])
        used_fac = set(assignments)
        total_open = np.sum(open_cost[list(used_fac)])
        return total_assign + total_open
        
    current_cost = calc_cost(result)
    
    # Iterative improvement
    improved = True
    max_iters = 50
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        for i in range(n_customers):
            current_fac = result[i]
            best_delta = 1e-9 # Threshold for improvement
            best_new_fac = -1
            
            for j in range(n_facilities):
                if j == current_fac:
                    continue
                
                # Check capacity
                new_load_j = load[j] + demands[i]
                if new_load_j > capacities[j]:
                    continue
                
                # Calculate delta
                delta_assign = assign_cost[i, j] - assign_cost[i, current_fac]
                delta_open = 0
                
                # Check if current_fac becomes empty
                old_load_current = load[current_fac]
                new_load_current = old_load_current - demands[i]
                
                if new_load_current == 0:
                    # Check if any other customer is assigned to current_fac
                    is_used = False
                    for k in range(n_customers):
                        if k != i and result[k] == current_fac:
                            is_used = True
                            break
                    if not is_used:
                        delta_open -= open_cost[current_fac]
                
                # Check if j was empty
                if load[j] == 0:
                    delta_open += open_cost[j]
                    
                total_delta = delta_assign + delta_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_new_fac = j
                    
            if best_new_fac != -1:
                # Perform move
                load[current_fac] -= demands[i]
                load[best_new_fac] += demands[i]
                result[i] = best_new_fac
                current_cost += best_delta
                improved = True
                
    return result

# EVOLVE-BLOCK-END
