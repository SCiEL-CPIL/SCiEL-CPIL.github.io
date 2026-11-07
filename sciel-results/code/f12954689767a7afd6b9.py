"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 120.0
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. LP Relaxation
    # Variables: x_0...x_{N-1} (N = n_customers * n_facilities), y_0...y_{n_f-1}
    n_vars = n_customers * n_facilities + n_facilities
    
    c_obj = np.zeros(n_vars)
    for i in range(n_customers):
        for j in range(n_facilities):
            c_obj[i * n_facilities + j] = assign_cost[i, j]
    for j in range(n_facilities):
        c_obj[n_customers * n_facilities + j] = open_cost[j]

    # Constraints:
    # 1. Sum_j x_ij = 1 for all i
    A_eq = np.zeros((n_customers, n_vars))
    b_eq = np.ones(n_customers)
    for i in range(n_customers):
        for j in range(n_facilities):
            A_eq[i, i * n_facilities + j] = 1.0

    # 2. Sum_i d_i x_ij - C_j y_j <= 0 for all j
    A_ub = np.zeros((n_facilities, n_vars))
    b_ub = np.zeros(n_facilities)
    for j in range(n_facilities):
        for i in range(n_customers):
            A_ub[j, i * n_facilities + j] = demands[i]
        A_ub[j, n_customers * n_facilities + j] = -capacities[j]
        b_ub[j] = 0

    bounds = [(0, 1)] * n_vars

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback to greedy
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        return result

    x_sol = res.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
    y_sol = res.x[n_customers * n_facilities:]

    # 2. Rounding
    # Identify open facilities based on LP solution
    # Threshold 0.5 is standard
    open_facilities = set(np.where(y_sol > 0.5)[0].tolist())
    
    if not open_facilities:
        # If none, open the cheapest one
        open_facilities.add(int(np.argmin(open_cost)))

    # Assign customers
    order = np.argsort(-demands)
    assignment = np.full(n_customers, -1, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in order:
        # Find best facility in current open set
        best_f = -1
        best_cost = float('inf')
        for f in open_facilities:
            if load[f] + demands[i] <= capacities[f]:
                if assign_cost[i, f] < best_cost:
                    best_cost = assign_cost[i, f]
                    best_f = f
        
        if best_f == -1:
            # Need to open a new facility
            best_f_new = -1
            best_total_cost = float('inf')
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    # Cost is assign cost + open cost if not already open
                    cost = assign_cost[i, f]
                    if f not in open_facilities:
                        cost += open_cost[f]
                    if cost < best_total_cost:
                        best_total_cost = cost
                        best_f_new = f
            
            if best_f_new != -1:
                best_f = best_f_new
                open_facilities.add(best_f)
        
        if best_f != -1:
            assignment[i] = best_f
            load[best_f] += demands[i]
        else:
            # Should not happen if total capacity >= total demand
            # Fallback to first facility with space
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    assignment[i] = f
                    load[f] += demands[i]
                    break

    # 3. Local Search
    def calc_cost(assignment):
        load = np.zeros(n_facilities)
        cost = 0.0
        used = np.zeros(n_facilities, dtype=bool)
        for i in range(n_customers):
            f = assignment[i]
            load[f] += demands[i]
            cost += assign_cost[i, f]
            used[f] = True
        if np.any(load > capacities + 1e-9):
            return float('inf')
        cost += np.sum(open_cost[used])
        return cost

    current_cost = calc_cost(assignment)
    
    # Precompute facility lists for swaps
    def get_facility_groups(assignment):
        groups = {f: [] for f in range(n_facilities)}
        for i in range(n_customers):
            groups[assignment[i]].append(i)
        return groups

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1-moves
        for i in range(n_customers):
            if time.time() > deadline: break
            current_f = assignment[i]
            for j in range(n_facilities):
                if j == current_f: continue
                if load[j] + demands[i] > capacities[j]:
                    continue
                
                delta_assign = assign_cost[i, j] - assign_cost[i, current_f]
                
                new_load_cf = load[current_f] - demands[i]
                new_load_j = load[j] + demands[i]
                
                delta_open = 0.0
                if new_load_cf == 0:
                    delta_open -= open_cost[current_f]
                if load[j] == 0:
                    delta_open += open_cost[j]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    load[current_f] -= demands[i]
                    load[j] += demands[i]
                    assignment[i] = j
                    current_cost += total_delta
                    improved = True
        
        # 2-swaps
        if time.time() < deadline - 5:
            groups = get_facility_groups(assignment)
            # Iterate over pairs of facilities
            for f1 in range(n_facilities):
                if time.time() > deadline: break
                if not groups[f1]: continue
                for f2 in range(f1 + 1, n_facilities):
                    if time.time() > deadline: break
                    if not groups[f2]: continue
                    
                    # Check if swap is possible capacity-wise
                    # We need to check for each pair i in f1, k in f2
                    # load[f1] - d_i + d_k <= C_f1
                    # load[f2] - d_k + d_i <= C_f2
                    
                    for i in groups[f1]:
                        if time.time() > deadline: break
                        for k in groups[f2]:
                            if time.time() > deadline: break
                            
                            # Capacity check
                            if load[f1] - demands[i] + demands[k] > capacities[f1]:
                                continue
                            if load[f2] - demands[k] + demands[i] > capacities[f2]:
                                continue
                            
                            delta_assign = (assign_cost[i, f2] + assign_cost[k, f1]) - (assign_cost[i, f1] + assign_cost[k, f2])
                            
                            # Open cost change
                            # f1 becomes empty if load[f1] == demands[i]
                            # f2 becomes empty if load[f2] == demands[k]
                            # f1 becomes non-empty? No, it was non-empty.
                            # f2 becomes non-empty? No.
                            
                            delta_open = 0.0
                            if load[f1] == demands[i]:
                                delta_open -= open_cost[f1]
                            if load[f2] == demands[k]:
                                delta_open -= open_cost[f2]
                            
                            # Note: If f1 or f2 becomes empty, we save cost.
                            # They never become non-empty in a swap between two non-empty facilities.
                            
                            total_delta = delta_assign + delta_open
                            
                            if total_delta < -1e-9:
                                # Perform swap
                                assignment[i] = f2
                                assignment[k] = f1
                                load[f1] = load[f1] - demands[i] + demands[k]
                                load[f2] = load[f2] - demands[k] + demands[i]
                                current_cost += total_delta
                                improved = True
                                # Update groups? 
                                # For simplicity, we break inner loops and restart search
                                # Or just continue, but groups are stale.
                                # Restarting is safer.
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    return assignment
# EVOLVE-BLOCK-END
