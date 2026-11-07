import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.array(demands)
    capacities = np.array(capacities)
    open_cost = np.array(open_cost)
    assign_cost = np.array(assign_cost)
    
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    count = np.zeros(n_facilities, dtype=int)
    
    # Greedy Initialization
    order = np.argsort(-demands)
    for i in order:
        feasible = capacities - load
        costs = assign_cost[i].copy()
        costs[feasible < demands[i]] = np.inf
        f_min = np.argmin(costs)
        
        if np.isinf(costs[f_min]):
            # Fallback: find any feasible
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    f_min = f
                    break
        
        assignment[i] = f_min
        load[f_min] += demands[i]
        count[f_min] += 1
        
    def compute_total_cost():
        cost = 0.0
        for f in range(n_facilities):
            if count[f] > 0:
                cost += open_cost[f]
        cost += np.sum(assign_cost[np.arange(n_customers), assignment])
        return cost

    current_cost = compute_total_cost()
    
    start_time = time.time()
    time_limit = 115.0
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # 1. Single Moves
        restart = True
        while restart and (time.time() - start_time) < time_limit:
            restart = False
            for i in range(n_customers):
                f_old = assignment[i]
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new]:
                        continue
                    
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    
                    delta_open = 0.0
                    if count[f_old] == 1:
                        delta_open -= open_cost[f_old]
                    if count[f_new] == 0:
                        delta_open += open_cost[f_new]
                    
                    if delta_assign + delta_open < -1e-9:
                        assignment[i] = f_new
                        load[f_old] -= demands[i]
                        count[f_old] -= 1
                        load[f_new] += demands[i]
                        count[f_new] += 1
                        current_cost += delta_assign + delta_open
                        improved = True
                        restart = True
                        break
        
        # 2. Swap Moves
        restart = True
        while restart and (time.time() - start_time) < time_limit:
            restart = False
            for i in range(n_customers):
                f_i = assignment[i]
                for j in range(i + 1, n_customers):
                    f_j = assignment[j]
                    if f_i == f_j:
                        continue
                    
                    delta = assign_cost[i, f_j] + assign_cost[j, f_i] - assign_cost[i, f_i] - assign_cost[j, f_j]
                    
                    if delta < -1e-9:
                        assignment[i] = f_j
                        assignment[j] = f_i
                        current_cost += delta
                        improved = True
                        restart = True
                        break
                if restart:
                    break

        # 3. Facility Closure
        for f_close in range(n_facilities):
            if count[f_close] == 0:
                continue
            
            customers_in_f = [i for i in range(n_customers) if assignment[i] == f_close]
            customers_in_f.sort(key=lambda x: -demands[x])
            
            temp_load = load.copy()
            temp_load[f_close] = 0
            temp_count = count.copy()
            temp_count[f_close] = 0
            temp_assignment = assignment.copy()
            
            feasible = True
            new_assign_cost = 0.0
            old_assign_cost = 0.0
            
            for i in customers_in_f:
                old_assign_cost += assign_cost[i, f_close]
                best_f = -1
                min_cost = np.inf
                for f_new in range(n_facilities):
                    if f_new == f_close:
                        continue
                    if temp_load[f_new] + demands[i] <= capacities[f_new]:
                        c = assign_cost[i, f_new]
                        if c < min_cost:
                            min_cost = c
                            best_f = f_new
                            
                if best_f == -1:
                    feasible = False
                    break
                else:
                    temp_load[best_f] += demands[i]
                    temp_count[best_f] += 1
                    temp_assignment[i] = best_f
                    new_assign_cost += min_cost
            
            if feasible:
                delta_open = -open_cost[f_close]
                for f_new in range(n_facilities):
                    if f_new == f_close:
                        continue
                    if temp_count[f_new] > 0 and count[f_new] == 0:
                        delta_open += open_cost[f_new]
                        
                total_delta = new_assign_cost - old_assign_cost + delta_open
                
                if total_delta < -1e-9:
                    assignment = temp_assignment
                    load = temp_load
                    count = temp_count
                    current_cost += total_delta
                    improved = True
                    break

    return assignment
