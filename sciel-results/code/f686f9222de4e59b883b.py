import numpy as np
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # LP Formulation
    # Variables: x[i, f] (n_cust * n_fac), y[f] (n_fac)
    # Total vars: n_cust * n_fac + n_fac
    n_vars = n_cust * n_fac + n_fac
    
    # Objective coefficients
    c = np.zeros(n_vars)
    for f in range(n_fac):
        # x[i, f] cost
        c[f * n_cust : (f + 1) * n_cust] = assign_cost[:, f]
        # y[f] cost
        c[n_cust * n_fac + f] = open_cost[f]
        
    # Equality constraints: sum_f x[i, f] = 1
    A_eq = np.zeros((n_cust, n_vars))
    b_eq = np.ones(n_cust)
    for i in range(n_cust):
        for f in range(n_fac):
            A_eq[i, f * n_cust + i] = 1.0
            
    # Inequality constraints:
    # 1. Capacity: sum_i demands[i] * x[i, f] <= capacities[f] * y[f]
    #    sum_i demands[i] * x[i, f] - capacities[f] * y[f] <= 0
    A_ub1 = np.zeros((n_fac, n_vars))
    b_ub1 = np.zeros(n_fac)
    for f in range(n_fac):
        for i in range(n_cust):
            A_ub1[f, f * n_cust + i] = demands[i]
        A_ub1[f, n_cust * n_fac + f] = -capacities[f]
        
    # 2. Linking: x[i, f] <= y[f]
    #    x[i, f] - y[f] <= 0
    A_ub2 = np.zeros((n_cust * n_fac, n_vars))
    b_ub2 = np.zeros(n_cust * n_fac)
    idx = 0
    for i in range(n_cust):
        for f in range(n_fac):
            A_ub2[idx, f * n_cust + i] = 1.0
            A_ub2[idx, n_cust * n_fac + f] = -1.0
            idx += 1
            
    A_ub = np.vstack([A_ub1, A_ub2])
    b_ub = np.concatenate([b_ub1, b_ub2])
    
    bounds = [(0, None)] * n_vars
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='HiGHS')
        if res.success:
            x_flat = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
            y = res.x[n_cust * n_fac:]
            
            # Check for integer solution
            if np.allclose(x_flat, np.round(x_flat), atol=1e-5) and np.allclose(y, np.round(y), atol=1e-5):
                assignment = np.argmax(x_flat, axis=1)
                return assignment.astype(int)
                
            # If not integer, use LP solution to initialize local search
            # Sort customers by demand descending
            cust_order = np.argsort(-demands)
            load = np.zeros(n_fac)
            assignment = np.zeros(n_cust, dtype=int)
            
            # Use LP fractional values to determine preference
            # For each customer, sort facilities by x[i, f] descending? 
            # Or just use assign_cost?
            # The LP already considers assign_cost and open_cost.
            # Let's use the LP's implied preference: for customer i, facility f is preferred if x[i,f] is high.
            # However, x[i,f] sums to 1.
            # Let's just use assign_cost for the greedy part, but open facilities based on y.
            
            open_facilities = [f for f in range(n_fac) if y[f] > 0.5]
            if not open_facilities:
                open_facilities = [np.argmin(open_cost)]
                
            for i in cust_order:
                best_f = -1
                best_cost = np.inf
                # Prefer facilities with higher y[i] in LP?
                # Let's sort open_facilities by assign_cost[i, f]
                for f in sorted(open_facilities, key=lambda f: assign_cost[i, f]):
                    if load[f] + demands[i] <= capacities[f]:
                        best_f = f
                        break
                if best_f == -1:
                    # Need to open a new facility
                    # Choose cheapest facility that can accommodate
                    candidates = [f for f in range(n_fac) if f not in open_facilities and load[f] + demands[i] <= capacities[f]]
                    if candidates:
                        best_f = min(candidates, key=lambda f: open_cost[f] + assign_cost[i, f])
                        open_facilities.append(best_f)
                    else:
                        # Fallback: any facility with capacity
                        for f in range(n_fac):
                            if load[f] + demands[i] <= capacities[f]:
                                best_f = f
                                break
                if best_f != -1:
                    assignment[i] = best_f
                    load[best_f] += demands[i]
                else:
                    # Infeasible, assign to facility with most capacity
                    remaining = capacities - load
                    best_f = np.argmax(remaining)
                    assignment[i] = best_f
                    load[best_f] += demands[i]
                    
            # Local Search
            assignment = _local_search(demands, capacities, open_cost, assign_cost, assignment)
            return assignment
    except Exception:
        pass
        
    # Fallback: Greedy
    return _greedy_assign(demands, capacities, open_cost, assign_cost)

def _greedy_assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    return result

def _local_search(demands, capacities, open_cost, assign_cost, assignment):
    n_cust = len(demands)
    n_fac = len(capacities)
    current_assignment = assignment.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[current_assignment[i]] += demands[i]
        
    def calc_cost(assignment, load):
        total_assign = 0
        open_fac = set()
        for i in range(n_cust):
            f = assignment[i]
            total_assign += assign_cost[i, f]
            open_fac.add(f)
        total_open = 0
        for f in open_fac:
            total_open += open_cost[f]
        return total_assign + total_open
        
    current_cost = calc_cost(current_assignment, load)
    
    improved = True
    max_iter = 50
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        # 1-Move
        for i in range(n_cust):
            f1 = current_assignment[i]
            for f2 in range(n_fac):
                if f1 == f2:
                    continue
                if load[f2] + demands[i] > capacities[f2]:
                    continue
                
                delta_assign = assign_cost[i, f2] - assign_cost[i, f1]
                delta_open = 0
                
                new_load_f1 = load[f1] - demands[i]
                new_load_f2 = load[f2] + demands[i]
                
                if new_load_f1 == 0:
                    delta_open -= open_cost[f1]
                if new_load_f2 == 0:
                    # This check is wrong. new_load_f2 == 0 is impossible if demands[i] > 0.
                    # We need to check if f2 was empty BEFORE the move.
                    pass
                
                # Correct open cost logic:
                # f1 was open. If new_load_f1 == 0, f1 closes. Save open_cost[f1].
                # f2 was open? If load[f2] == 0, f2 was closed. If we assign i, f2 opens. Pay open_cost[f2].
                if load[f2] == 0:
                    delta_open += open_cost[f2]
                    
                if delta_assign + delta_open < 0:
                    current_assignment[i] = f2
                    load[f1] -= demands[i]
                    load[f2] += demands[i]
                    current_cost += delta_assign + delta_open
                    improved = True
                    
        # 2-Swap
        # Swap customer i from f1 and customer j from f2
        # Condition: f1 != f2
        # New assignment: i->f2, j->f1
        # Capacity: load[f1] - d_i + d_j <= cap[f1] AND load[f2] - d_j + d_i <= cap[f2]
        # Cost delta:
        # assign: (d_i,f2 + d_j,f1) - (d_i,f1 + d_j,f2)
        # open: 
        #   f1 was open. New load: load[f1] - d_i + d_j. If 0, close.
        #   f2 was open. New load: load[f2] - d_j + d_i. If 0, close.
        
        for i in range(n_cust):
            f1 = current_assignment[i]
            di = demands[i]
            for j in range(i + 1, n_cust):
                f2 = current_assignment[j]
                if f1 == f2:
                    continue
                dj = demands[j]
                
                new_load_f1 = load[f1] - di + dj
                new_load_f2 = load[f2] - dj + di
                
                if new_load_f1 > capacities[f1] or new_load_f2 > capacities[f2]:
                    continue
                
                delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])
                
                delta_open = 0
                # f1
                if load[f1] == di: # Only i was at f1
                    if new_load_f1 == 0: # Impossible since dj > 0
                        pass
                    else:
                        # f1 remains open
                        pass
                else:
                    # f1 had others
                    if new_load_f1 == 0: # Impossible
                        pass
                    
                # Simpler open cost check:
                # Old open set: {f1, f2} (plus others)
                # New open set: 
                # f1 is open if new_load_f1 > 0
                # f2 is open if new_load_f2 > 0
                
                old_f1_open = 1 if load[f1] > 0 else 0
                old_f2_open = 1 if load[f2] > 0 else 0
                new_f1_open = 1 if new_load_f1 > 0 else 0
                new_f2_open = 1 if new_load_f2 > 0 else 0
                
                if old_f1_open and not new_f1_open:
                    delta_open -= open_cost[f1]
                if not old_f1_open and new_f1_open:
                    delta_open += open_cost[f1]
                if old_f2_open and not new_f2_open:
                    delta_open -= open_cost[f2]
                if not old_f2_open and new_f2_open:
                    delta_open += open_cost[f2]
                    
                if delta_assign + delta_open < 0:
                    current_assignment[i] = f2
                    current_assignment[j] = f1
                    load[f1] = new_load_f1
                    load[f2] = new_load_f2
                    current_cost += delta_assign + delta_open
                    improved = True
                    
    return current_assignment
