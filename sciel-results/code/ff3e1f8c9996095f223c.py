import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds
import warnings

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 130.0
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Variables:
    # 0 to n_customers*n_facilities - 1 : x_{i,f}
    # n_customers*n_facilities to n_customers*n_facilities + n_facilities - 1 : y_f
    
    n_x = n_customers * n_facilities
    n_y = n_facilities
    n_vars = n_x + n_y
    
    # Objective: min sum(open_cost[f] * y_f) + sum(assign_cost[i,f] * x_{i,f})
    c = np.zeros(n_vars)
    
    # y_f coefficients
    for f in range(n_facilities):
        c[n_x + f] = open_cost[f]
        
    # x_{i,f} coefficients
    for i in range(n_customers):
        for f in range(n_facilities):
            c[i * n_facilities + f] = assign_cost[i, f]
            
    # Bounds: all binary
    integrality = np.ones(n_vars)
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Constraints:
    # 1. For each customer i: sum_f x_{i,f} = 1
    # 2. For each facility f: sum_i demands[i] * x_{i,f} - capacities[f] * y_f <= 0
    
    A_rows = []
    lb_rows = []
    ub_rows = []
    
    # Customer assignment constraints
    for i in range(n_customers):
        row = np.zeros(n_vars)
        for f in range(n_facilities):
            row[i * n_facilities + f] = 1.0
        A_rows.append(row)
        lb_rows.append(1.0)
        ub_rows.append(1.0)
        
    # Facility capacity constraints
    for f in range(n_facilities):
        row = np.zeros(n_vars)
        for i in range(n_customers):
            row[i * n_facilities + f] = demands[i]
        row[n_x + f] = -capacities[f]
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
        
    A = np.array(A_rows)
    lb = np.array(lb_rows)
    ub = np.array(ub_rows)
    
    constraint = LinearConstraint(A, lb, ub)
    bounds = Bounds(lb, ub)
    
    try:
        result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if result.success and result.x is not None:
            x_flat = result.x[:n_x]
            # Rounding might be needed if solver returns non-integer due to tolerance, 
            # but integrality=1 should enforce it.
            x_matrix = x_flat.reshape(n_customers, n_facilities)
            
            # Determine assignment: argmax of x_matrix (should be 1.0)
            assignment = np.argmax(x_matrix, axis=1).astype(int)
            
            # Verify validity
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[assignment[i]] += demands[i]
            
            if np.all(load <= capacities + 1e-6):
                return assignment
            else:
                # If invalid, fallback to heuristic
                pass
        else:
            # Fallback
            pass
            
    except Exception as e:
        # Fallback
        pass

    # Fallback Heuristic: Greedy + Local Search
    # This is similar to the original but we can make it slightly more robust
    # if MILP fails.
    
    def greedy_seed(order_type='desc'):
        res = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        if order_type == 'desc':
            order = np.argsort(-demands)
        else:
            order = np.argsort(demands)
            
        for i in order:
            # Sort facilities by cost for this customer
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    res[i] = f
                    break
            # If no facility found, pick cheapest feasible ignoring capacity? 
            # No, must be valid. If greedy fails, try random feasible.
            if load[res[i]] == 0 and res[i] == 0 and load[0] + demands[i] > capacities[0]:
                 # Check if 0 was actually assigned
                 pass
                 
        # Verify if all assigned
        for i in range(n_customers):
             if load[res[i]] < demands[i]: # This check is tricky with the loop above
                 pass
                 
        # A safer greedy:
        res = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        for i in order:
            feasible = [f for f in fac_order if load[f] + demands[i] <= capacities[f]]
            if feasible:
                # Pick cheapest feasible
                f_best = min(feasible, key=lambda f: assign_cost[i, f])
                res[i] = f_best
                load[f_best] += demands[i]
            else:
                # Should not happen if total capacity > total demand
                # Force assign to one with most space
                f_best = np.argmax(capacities - load)
                res[i] = f_best
                load[f_best] += demands[i]
                
        return res
        
    def local_search(assignment):
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
            
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n_customers):
                curr_f = assignment[i]
                # Try moving i to any other facility
                # Calculate delta for moving i from curr_f to new_f
                # Cost change: assign_cost[i, new_f] - assign_cost[i, curr_f]
                # Open cost change:
                #   if load[curr_f] == demands[i] and no one else at curr_f: -open_cost[curr_f]
                #   if load[new_f] == 0: +open_cost[new_f]
                
                curr_load = load[curr_f]
                
                for new_f in range(n_facilities):
                    if new_f == curr_f:
                        continue
                    if load[new_f] + demands[i] > capacities[new_f]:
                        continue
                        
                    delta = assign_cost[i, new_f] - assign_cost[i, curr_f]
                    
                    # Open cost logic
                    # If curr_f becomes empty
                    if curr_load == demands[i]:
                         # Check if any other customer is at curr_f
                         others = np.sum(assignment == curr_f)
                         if others == 1: # Only i is there
                             delta -= open_cost[curr_f]
                             
                    # If new_f becomes non-empty
                    if load[new_f] == 0:
                         delta += open_cost[new_f]
                         
                    if delta < -1e-9:
                         # Accept
                         load[curr_f] -= demands[i]
                         load[new_f] += demands[i]
                         assignment[i] = new_f
                         improved = True
                         break # Restart loop for this i? Or continue? 
                         # Breaking inner loop means we move i and don't try other new_f for i in this pass.
                         
        return assignment
        
    best_assign = greedy_seed('desc')
    best_assign = local_search(best_assign)
    
    # Try another seed
    if time.time() < deadline - 20:
        alt_assign = greedy_seed('asc')
        alt_assign = local_search(alt_assign)
        
        # Compare costs
        def calc_cost(a):
             cost = 0.0
             for f in range(n_facilities):
                 if np.any(a == f):
                     cost += open_cost[f]
             cost += np.sum(assign_cost[np.arange(n_customers), a])
             return cost
             
        if calc_cost(alt_assign) < calc_cost(best_assign):
            best_assign = alt_assign
            
    return best_assign
