import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.array(demands)
    capacities = np.array(capacities)
    open_cost = np.array(open_cost)
    assign_cost = np.array(assign_cost)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Initial Solution: Greedy
    idx_sorted = np.argsort(-demands)
    
    x = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in idx_sorted:
        best_f = -1
        best_cost = float('inf')
        for j in range(n_facilities):
            if load[j] + demands[i] <= capacities[j]:
                c = assign_cost[i][j]
                if load[j] == 0:
                    c += open_cost[j]
                if c < best_cost:
                    best_cost = c
                    best_f = j
        if best_f != -1:
            x[i] = best_f
            load[best_f] += demands[i]
        else:
            for j in range(n_facilities):
                if load[j] + demands[i] <= capacities[j]:
                    x[i] = j
                    load[j] += demands[i]
                    break
    
    def calc_cost(x, load):
        cost = 0.0
        for j in range(n_facilities):
            if load[j] > 0:
                cost += open_cost[j]
        cost += np.sum(assign_cost[np.arange(n_customers), x])
        return cost

    current_cost = calc_cost(x, load)
    best_x = x.copy()
    best_load = load.copy()
    best_cost = current_cost
    
    # 2. Simulated Annealing
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    T = 5.0  # Initial temperature
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(n_customers)
        old_f = x[i]
        
        # Try a random different facility
        j = np.random.randint(n_facilities)
        if j == old_f:
            continue
            
        if load[j] + demands[i] > capacities[j]:
            continue
            
        delta_assign = assign_cost[i][j] - assign_cost[i][old_f]
        delta_open = 0.0
        
        if load[old_f] == demands[i]:
            delta_open -= open_cost[old_f]
            
        if load[j] == 0:
            delta_open += open_cost[j]
            
        delta_total = delta_assign + delta_open
        
        if delta_total < 0 or np.random.rand() < np.exp(-delta_total / T):
            x[i] = j
            load[old_f] -= demands[i]
            load[j] += demands[i]
            current_cost += delta_total
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_x = x.copy()
                best_load = load.copy()
                
        T = max(T * cooling, T_min)
        
    return best_x
