"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    timeout = 145.0  # Safety margin below 150s
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    def time_exceeded():
        return time.time() - start_time > timeout

    # 1. Solve LP Relaxation
    # Variables: x_{ij} (N*M), y_j (M)
    n_vars = n_customers * n_facilities + n_facilities
    
    c = np.zeros(n_vars)
    c[:n_customers * n_facilities] = assign_cost.flatten()
    c[n_customers * n_facilities:] = open_cost
    
    # Relax integrality: all continuous
    integrality = np.zeros(n_vars)
    
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    A = np.zeros((n_customers + n_facilities, n_vars))
    lb = np.zeros(n_customers + n_facilities)
    ub = np.zeros(n_customers + n_facilities)
    
    for i in range(n_customers):
        for j in range(n_facilities):
            A[i, i * n_facilities + j] = 1.0
        lb[i] = 1.0
        ub[i] = 1.0
        
    for j in range(n_facilities):
        for i in range(n_customers):
            A[n_customers + j, i * n_facilities + j] = demands[i]
        A[n_customers + j, n_customers * n_facilities + j] = -capacities[j]
        lb[n_customers + j] = -np.inf
        ub[n_customers + j] = 0.0
        
    constraints = LinearConstraint(A, lb, ub)
    
    x_frac = None
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={'time_limit': 10.0})
        if res.success:
            x_frac = res.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
    except Exception:
        pass

    # 2. Rounding Procedure
    def round_solution(x_frac):
        if x_frac is None:
            # Fallback: Greedy by cost
            res = np.zeros(n_customers, dtype=int)
            load = np.zeros(n_facilities, dtype=float)
            order = np.argsort(-demands)
            for i in order:
                d = demands[i]
                remaining_cap = capacities - load
                feasible_mask = remaining_cap >= d
                if not np.any(feasible_mask):
                    f = np.argmax(remaining_cap)
                else:
                    costs = np.where(feasible_mask, assign_cost[i], np.inf)
                    f = np.argmin(costs)
                res[i] = f
                load[f] += d
            return res
        
        # Sort customers by demand descending
        order = np.argsort(-demands)
        assignment = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=float)
        
        for i in order:
            d = demands[i]
            remaining_cap = capacities - load
            feasible_mask = remaining_cap >= d
            
            if not np.any(feasible_mask):
                # No feasible facility with capacity, pick one with max remaining cap
                f = np.argmax(remaining_cap)
            else:
                # Among feasible, pick the one with highest x_frac[i, j]
                # If x_frac is not available or all zero, fall back to min cost
                scores = np.where(feasible_mask, x_frac[i], -np.inf)
                if np.all(scores == -np.inf):
                    costs = np.where(feasible_mask, assign_cost[i], np.inf)
                    f = np.argmin(costs)
                else:
                    f = np.argmax(scores)
            assignment[i] = f
            load[f] += d
            
        return assignment

    current_assignment = round_solution(x_frac)
    
    # 3. Local Search
    def calculate_cost(assignment):
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        
        # Check validity
        if np.any(load > capacities + 1e-6):
            return np.inf
            
        assign_sum = np.sum(assign_cost[np.arange(n_customers), assignment])
        open_sum = np.sum(open_cost[load > 0])
        return assign_sum + open_sum

    current_cost = calculate_cost(current_assignment)
    
    if np.isinf(current_cost):
        # If rounding failed to produce a valid solution, use greedy fallback
        res = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=float)
        order = np.argsort(-demands)
        for i in order:
            d = demands[i]
            remaining_cap = capacities - load
            feasible_mask = remaining_cap >= d
            if not np.any(feasible_mask):
                f = np.argmax(remaining_cap)
            else:
                costs = np.where(feasible_mask, assign_cost[i], np.inf)
                f = np.argmin(costs)
            res[i] = f
            load[f] += d
        return res

    while not time_exceeded():
        improved = False
        
        # 1-move local search
        for i in range(n_customers):
            if time_exceeded():
                break
            curr_f = current_assignment[i]
            curr_cost_i = assign_cost[i, curr_f]
            
            for new_f in range(n_facilities):
                if new_f == curr_f:
                    continue
                
                # Check capacity
                load_new_f = np.sum(demands[current_assignment == new_f])
                if load_new_f + demands[i] > capacities[new_f] + 1e-6:
                    continue
                
                # Calculate delta
                new_cost_i = assign_cost[i, new_f]
                delta_assign = new_cost_i - curr_cost_i
                
                # Check if opening/closing costs change
                # If new_f was empty and becomes non-empty, add open_cost[new_f]
                # If curr_f becomes empty, subtract open_cost[curr_f]
                delta_open = 0
                load_curr_f = np.sum(demands[current_assignment == curr_f])
                load_new_f_before = load_new_f
                
                if load_new_f_before == 0 and load_curr_f > demands[i]:
                    # new_f was empty, now non-empty. curr_f stays non-empty.
                    delta_open = open_cost[new_f]
                elif load_curr_f == demands[i] and load_new_f_before > 0:
                    # curr_f becomes empty, new_f stays non-empty.
                    delta_open = -open_cost[curr_f]
                elif load_curr_f == demands[i] and load_new_f_before == 0:
                    # curr_f becomes empty, new_f becomes non-empty.
                    delta_open = open_cost[new_f] - open_cost[curr_f]
                # else: no change in open status
                
                delta = delta_assign + delta_open
                
                if delta < -1e-9:
                    current_assignment[i] = new_f
                    current_cost += delta
                    improved = True
                    break # Restart 1-move loop after a change to be safe, or just continue?
                    # Usually, continuing is fine, but restarting ensures we don't miss better moves
                    # However, for speed, we can just continue. Let's continue.
        
        # 2-swap local search
        for i in range(n_customers):
            if time_exceeded():
                break
            f_i = current_assignment[i]
            d_i = demands[i]
            
            for k in range(i + 1, n_customers):
                if time_exceeded():
                    break
                f_k = current_assignment[k]
                if f_i == f_k:
                    continue
                d_k = demands[k]
                
                # Check capacities
                # If we swap, f_i loses d_i and gains d_k. f_k loses d_k and gains d_i.
                load_f_i = np.sum(demands[current_assignment == f_i])
                load_f_k = np.sum(demands[current_assignment == f_k])
                
                if load_f_i - d_i + d_k > capacities[f_i] + 1e-6:
                    continue
                if load_f_k - d_k + d_i > capacities[f_k] + 1e-6:
                    continue
                
                # Calculate delta
                delta_assign = (assign_cost[i, f_k] + assign_cost[k, f_i]) - (assign_cost[i, f_i] + assign_cost[k, f_k])
                
                delta_open = 0
                # Check if facilities become empty
                if load_f_i == d_i and load_f_k > d_k:
                    delta_open = -open_cost[f_i]
                elif load_f_k == d_k and load_f_i > d_i:
                    delta_open = -open_cost[f_k]
                elif load_f_i == d_i and load_f_k == d_k:
                    delta_open = -(open_cost[f_i] + open_cost[f_k])
                # else: no change in open status (both stay non-empty)
                
                delta = delta_assign + delta_open
                
                if delta < -1e-9:
                    current_assignment[i] = f_k
                    current_assignment[k] = f_i
                    current_cost += delta
                    improved = True
                    # Break out of inner loop? Or just update and continue?
                    # Breaking inner loop is safer to recalculate loads if we do it naively.
                    # But we can just continue.
                    break
        
        if not improved:
            break
            
    return current_assignment
# EVOLVE-BLOCK-END
