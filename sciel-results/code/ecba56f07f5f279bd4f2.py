import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    N = len(demands)
    M = len(capacities)
    
    # 1. Greedy Initialization
    # Sort customers by demand descending
    order = np.argsort(-demands)
    current_solution = np.zeros(N, dtype=int)
    load = np.zeros(M)
    
    # Pre-compute sorted facilities for each customer by cost
    # To speed up greedy, we can just iterate facilities
    for i in order:
        # Find cheapest facility with capacity
        # We want min assign_cost[i, f] where load[f] + demands[i] <= capacities[f]
        # Mask out infeasible
        feasible_mask = load + demands[i] <= capacities
        if not np.any(feasible_mask):
            # Should not happen if total cap >= total demand
            # Fallback: pick any with max remaining cap? Or raise error
            # For robustness, pick facility with most remaining capacity
            idx = np.argmax(capacities - load)
            current_solution[i] = idx
            load[idx] += demands[i]
            continue
            
        # Set infeasible costs to inf
        costs = assign_cost[i].copy()
        costs[~feasible_mask] = np.inf
        f = np.argmin(costs)
        current_solution[i] = f
        load[f] += demands[i]
        
    # Helper to calculate total cost
    def calc_cost(sol, loads):
        # Open costs
        open_mask = loads > 0
        cost = np.sum(open_cost[open_mask])
        # Assign costs
        assign_sum = 0
        for i in range(N):
            assign_sum += assign_cost[i, sol[i]]
        return cost + assign_sum

    current_cost = calc_cost(current_solution, load)
    best_solution = current_solution.copy()
    best_cost = current_cost
    start_time = time.time()
    limit = 140.0 # Safety margin
    
    # Local Search
    while time.time() - start_time < limit:
        improved = False
        
        # --- Single Move Phase ---
        # We do best-improvement
        best_delta = 0
        best_move = None
        for i in range(N):
            curr_f = current_solution[i]
            for k in range(M):
                if k == curr_f:
                    continue
                if load[k] + demands[i] > capacities[k]:
                    continue
                
                # Calculate Delta
                # Assign cost change
                d_assign = assign_cost[i, k] - assign_cost[i, curr_f]
                
                # Open cost change
                d_open = 0
                new_load_curr = load[curr_f] - demands[i]
                if new_load_curr == 0:
                    d_open -= open_cost[curr_f]
                
                if load[k] == 0:
                    d_open += open_cost[k]
                    
                total_delta = d_assign + d_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_move = (i, k)
                    
        if best_move is not None:
            i, k = best_move
            curr_f = current_solution[i]
            # Apply
            load[curr_f] -= demands[i]
            load[k] += demands[i]
            current_solution[i] = k
            current_cost += best_delta
            improved = True
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution.copy()
                
        # --- Swap Phase ---
        best_delta = 0
        best_swap = None
        for i in range(N):
            f_i = current_solution[i]
            for m in range(i + 1, N):
                f_m = current_solution[m]
                if f_i == f_m:
                    continue
                
                # Check feasibility
                new_load_fi = load[f_i] - demands[i] + demands[m]
                new_load_fm = load[f_m] - demands[m] + demands[i]
                
                if new_load_fi > capacities[f_i] or new_load_fm > capacities[f_m]:
                    continue
                
                # Calculate Delta
                d_assign = (assign_cost[i, f_m] + assign_cost[m, f_i]) - (assign_cost[i, f_i] + assign_cost[m, f_m])
                
                d_open = 0
                # f_i was open. If new_load_fi == 0, it closes.
                if new_load_fi == 0:
                    d_open -= open_cost[f_i]
                # f_m was open. If new_load_fm == 0, it closes.
                if new_load_fm == 0:
                    d_open -= open_cost[f_m]
                
                total_delta = d_assign + d_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_swap = (i, m)
                    
        if best_swap is not None:
            i, m = best_swap
            f_i = current_solution[i]
            f_m = current_solution[m]
            
            # Apply
            load[f_i] -= demands[i]
            load[f_i] += demands[m]
            load[f_m] -= demands[m]
            load[f_m] += demands[i]
            
            current_solution[i] = f_m
            current_solution[m] = f_i
            
            current_cost += best_delta
            improved = True
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution.copy()
                
        if not improved:
            break
            
    return best_solution
