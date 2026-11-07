"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    timeout = 115.0  # Safety margin below 120s
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Convert to numpy arrays for ease of use
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    # 1. Solve as min-cost flow: each customer assigned to exactly one facility
    # Variables: x[i,f] = 1 if customer i assigned to facility f
    # Minimize: sum_i,f assign_cost[i,f] * x[i,f]
    # Subject to:
    #   For each i: sum_f x[i,f] = 1
    #   For each f: sum_i demands[i]*x[i,f] <= capacities[f]
    #   x[i,f] in {0,1} (relax to [0,1] for LP)
    
    # Build LP: n_customers * n_facilities variables
    n_vars = n_customers * n_facilities
    
    # Objective: minimize sum of assign_cost * x
    c = assign_cost.flatten()
    
    # Equality constraints: each customer assigned to exactly one facility
    # For customer i: sum_f x[i,f] = 1
    A_eq = np.zeros((n_customers, n_vars))
    for i in range(n_customers):
        for f in range(n_facilities):
            A_eq[i, i * n_facilities + f] = 1.0
    
    b_eq = np.ones(n_customers)
    
    # Inequality constraints: capacity constraints
    # For facility f: sum_i demands[i] * x[i,f] <= capacities[f]
    A_ub = np.zeros((n_facilities, n_vars))
    for f in range(n_facilities):
        for i in range(n_customers):
            A_ub[f, i * n_facilities + f] = demands[i]
    
    b_ub = capacities
    
    # Bounds: x in [0, 1]
    bounds = [(0, 1)] * n_vars
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status == 0:
            x = res.x.reshape(n_customers, n_facilities)
            # Round to nearest integer assignment (should be close to integral for this structure)
            result = np.zeros(n_customers, dtype=int)
            for i in range(n_customers):
                result[i] = np.argmax(x[i])
        else:
            # Fallback to greedy
            result = greedy_assign(demands, capacities, assign_cost)
    except:
        result = greedy_assign(demands, capacities, assign_cost)
    
    load = np.zeros(n_facilities, dtype=float)
    for i in range(n_customers):
        load[result[i]] += demands[i]
    
    # Verify feasibility, fix if needed
    result = fix_feasibility(result, demands, capacities)
    load = np.zeros(n_facilities, dtype=float)
    for i in range(n_customers):
        load[result[i]] += demands[i]
    
    def calc_total_cost(result, load):
        open_cost_sum = np.sum(open_cost[load > 0])
        assign_cost_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_cost_sum + assign_cost_sum

    current_cost = calc_total_cost(result, load)
    best_cost = current_cost
    best_result = result.copy()
    best_load = load.copy()
    
    # 2. Local Search: Best Improvement
    improved = True
    while improved:
        if time.time() - start_time > timeout:
            break
        improved = False
        best_move = None
        best_delta = 0.0
        
        for i in range(n_customers):
            f_old = result[i]
            d = demands[i]
            
            for f_new in range(n_facilities):
                if f_new == f_old:
                    continue
                
                if load[f_new] + d > capacities[f_new]:
                    continue
                
                delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                delta_open = 0.0
                
                if load[f_old] == d:
                    delta_open -= open_cost[f_old]
                
                if load[f_new] == 0:
                    delta_open += open_cost[f_new]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_move = (i, f_old, f_new)
        
        if best_move is not None:
            i, f_old, f_new = best_move
            load[f_old] -= demands[i]
            load[f_new] += demands[i]
            result[i] = f_new
            
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_result = result.copy()
                best_load = load.copy()
            improved = True
        else:
            break

    # 3. Second phase: multi-start local search with perturbation
    if time.time() - start_time < timeout - 10:
        np.random.seed(42)
        for restart in range(20):
            if time.time() - start_time > timeout:
                break
            
            # Start from best result and perturb
            result = best_result.copy()
            load = best_load.copy()
            
            # Perturb: randomly reassign some customers
            n_perturb = max(5, n_customers // 10)
            perm = np.random.permutation(n_customers)[:n_perturb]
            for i in perm:
                f_old = result[i]
                d = demands[i]
                feasible = [f for f in range(n_facilities) if f != f_old and load[f] + d <= capacities[f]]
                if feasible:
                    f_new = np.random.choice(feasible)
                    load[f_old] -= d
                    load[f_new] += d
                    result[i] = f_new
            
            current_cost = calc_total_cost(result, load)
            
            # Local search on perturbed solution
            for _ in range(50):  # Max local search iterations
                if time.time() - start_time > timeout:
                    break
                best_move = None
                best_delta = 0.0
                
                for i in range(n_customers):
                    f_old = result[i]
                    d = demands[i]
                    
                    for f_new in range(n_facilities):
                        if f_new == f_old:
                            continue
                        
                        if load[f_new] + d > capacities[f_new]:
                            continue
                        
                        delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                        delta_open = 0.0
                        
                        if load[f_old] == d:
                            delta_open -= open_cost[f_old]
                        
                        if load[f_new] == 0:
                            delta_open += open_cost[f_new]
                        
                        total_delta = delta_assign + delta_open
                        
                        if total_delta < best_delta:
                            best_delta = total_delta
                            best_move = (i, f_old, f_new)
                
                if best_move is None:
                    break
                
                i, f_old, f_new = best_move
                load[f_old] -= demands[i]
                load[f_new] += demands[i]
                result[i] = f_new
                current_cost += best_delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = result.copy()
                    best_load = load.copy()

    return best_result.astype(int)


def greedy_assign(demands, capacities, assign_cost):
    """Fallback greedy assignment."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    
    order = np.argsort(-demands)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=float)
    
    for i in order:
        d = demands[i]
        remaining_cap = capacities - load
        feasible_mask = remaining_cap >= d
        if not np.any(feasible_mask):
            f = np.argmax(remaining_cap)
        else:
            costs = np.where(feasible_mask, assign_cost[i], np.inf)
            f = np.argmin(costs)
        
        result[i] = f
        load[f] += d
    
    return result


def fix_feasibility(result, demands, capacities):
    """Fix any capacity violations by reassigning customers."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    
    result = result.copy()
    load = np.zeros(n_facilities, dtype=float)
    for i in range(n_customers):
        load[result[i]] += demands[i]
    
    # If any facility is over capacity, move customers to less loaded facilities
    for f in range(n_facilities):
        while load[f] > capacities[f]:
            # Find customer in f with smallest demand
            customers_in_f = [i for i in range(n_customers) if result[i] == f]
            if not customers_in_f:
                break
            i = min(customers_in_f, key=lambda x: demands[x])
            d = demands[i]
            
            # Find feasible facility with most remaining capacity
            best_f = None
            best_rem = -1
            for f2 in range(n_facilities):
                if f2 == f:
                    continue
                rem = capacities[f2] - load[f2]
                if rem >= d and rem > best_rem:
                    best_rem = rem
                    best_f = f2
            
            if best_f is None:
                # Can't fix, just assign to facility with most remaining capacity
                best_f = np.argmax(capacities - load)
            
            load[f] -= d
            load[best_f] += d
            result[i] = best_f
    
    return result
# EVOLVE-BLOCK-END
