import numpy as np
import time
import random
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    # --- Fallback Heuristic Logic (from current program, condensed) ---
    def compute_cost(result):
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(result):
            load[f] += demands[i]
            fac_count[f] += 1
        if np.any(load > capacities + 1e-9):
            return float('inf')
        return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), result])

    def greedy_init(order_func, randomize=False):
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(order_func(demands))
        
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            if randomize:
                perturbed = assign_cost[i] + np.random.uniform(0, 1) * 10
                fac_order = np.argsort(perturbed)
            
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            
            if not assigned:
                remaining = capacities - load
                f = np.argmax(remaining)
                if remaining[f] >= demands[i]:
                    load[f] += demands[i]
                    result[i] = f
                else:
                    return None
            
        return result

    # Generate a quick heuristic solution first as a safety net
    candidates = []
    init_g1 = greedy_init(lambda d: -d)
    if init_g1 is not None:
        candidates.append(init_g1)
        
    init_g2 = greedy_init(lambda d: d)
    if init_g2 is not None:
        candidates.append(init_g2)
        
    best_fallback = None
    if candidates:
        best_fallback = min(candidates, key=compute_cost)
    else:
        # Last resort: just fill sequentially
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in range(n_cust):
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        best_fallback = result

    # --- 1. Exact MILP Formulation ---
    # Variables:
    # x[i, j] = 1 if customer i is assigned to facility j (0 <= x <= 1, integer)
    # y[j] = 1 if facility j is open (0 <= y <= 1, integer)
    # Total variables: n_cust * n_fac + n_fac
    
    n_vars = n_cust * n_fac + n_fac
    
    # Objective: Minimize sum(assign_cost[i, j] * x[i, j]) + sum(open_cost[j] * y[j])
    c_obj = np.zeros(n_vars)
    for i in range(n_cust):
        for j in range(n_fac):
            c_obj[i * n_fac + j] = assign_cost[i, j]
    for j in range(n_fac):
        c_obj[n_cust * n_fac + j] = open_cost[j]
        
    # Integrality: All variables are binary (0 or 1)
    integrality = np.ones(n_vars)
    
    # Bounds: All variables are in [0, 1]
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility: sum_j x[i, j] = 1 for all i
    # 2. Capacity constraint: sum_i demands[i] * x[i, j] <= capacities[j] for all j
    # 3. Linking constraint: x[i, j] <= y[j] for all i, j (if assigned, facility must be open)
    
    # Let's build the constraint matrix A_ub for <= constraints and A_eq for = constraints
    # scipy.optimize.milp uses:
    # constraints: LinearConstraint(A, lb, ub)
    # where A @ x >= lb and A @ x <= ub is not directly supported as separate A_ub/A_eq in the simple interface?
    # Actually, scipy.optimize.milp takes a list of LinearConstraints or a single LinearConstraint.
    # LinearConstraint(A, lb, ub) means lb <= A @ x <= ub.
    
    # We need:
    # Eq: sum_j x[i, j] = 1  ->  1 <= sum_j x[i, j] <= 1
    # Ineq: sum_i demands[i] x[i, j] <= capacities[j] ->  -inf <= sum_i demands[i] x[i, j] <= capacities[j]
    # Ineq: x[i, j] - y[j] <= 0 ->  -inf <= x[i, j] - y[j] <= 0
    
    # Total rows in A:
    # n_cust (for assignment equality)
    # n_fac (for capacity)
    # n_cust * n_fac (for linking)
    
    n_rows = n_cust + n_fac + n_cust * n_fac
    A = np.zeros((n_rows, n_vars))
    lb_cons = np.zeros(n_rows)
    ub_cons = np.zeros(n_rows)
    
    # 1. Assignment Equality: sum_j x[i, j] = 1
    row_idx = 0
    for i in range(n_cust):
        for j in range(n_fac):
            A[row_idx, i * n_fac + j] = 1.0
        lb_cons[row_idx] = 1.0
        ub_cons[row_idx] = 1.0
        row_idx += 1
        
    # 2. Capacity: sum_i demands[i] x[i, j] <= capacities[j]
    for j in range(n_fac):
        for i in range(n_cust):
            A[row_idx, i * n_fac + j] = demands[i]
        lb_cons[row_idx] = -np.inf
        ub_cons[row_idx] = capacities[j]
        row_idx += 1
        
    # 3. Linking: x[i, j] <= y[j]  =>  x[i, j] - y[j] <= 0
    for i in range(n_cust):
        for j in range(n_fac):
            A[row_idx, i * n_fac + j] = 1.0
            A[row_idx, n_cust * n_fac + j] = -1.0
            lb_cons[row_idx] = -np.inf
            ub_cons[row_idx] = 0.0
            row_idx += 1
            
    constraints = LinearConstraint(A, lb_cons, ub_cons)
    bounds = Bounds(lb, ub)
    
    milp_solution = None
    try:
        if time.time() < deadline - 5:
            # Use HiGHS solver
            res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds, options={'presolve': True, 'time_limit': max(0, deadline - time.time() - 2)})
            if res.status == 0: # Optimal
                x_sol = res.x
                # Extract assignment
                x_assign = x_sol[:n_cust * n_fac].reshape(n_cust, n_fac)
                # Round to nearest integer (should be close to 0 or 1)
                x_assign_rounded = np.round(x_assign).astype(int)
                
                # Verify feasibility
                valid = True
                load = np.zeros(n_fac)
                for i in range(n_cust):
                    assigned_facs = np.where(x_assign_rounded[i] == 1)[0]
                    if len(assigned_facs) != 1:
                        valid = False
                        break
                    f = assigned_facs[0]
                    load[f] += demands[i]
                
                if np.any(load > capacities + 1e-9):
                    valid = False
                    
                if valid:
                    # Construct result
                    result = np.zeros(n_cust, dtype=int)
                    for i in range(n_cust):
                        result[i] = np.where(x_assign_rounded[i] == 1)[0][0]
                    milp_solution = result
            elif res.status == 1: # Iteration limit or other
                # Check if a feasible solution was found
                if hasattr(res, 'x') and res.x is not None:
                    x_sol = res.x
                    x_assign = x_sol[:n_cust * n_fac].reshape(n_cust, n_fac)
                    x_assign_rounded = np.round(x_assign).astype(int)
                    
                    valid = True
                    load = np.zeros(n_fac)
                    for i in range(n_cust):
                        assigned_facs = np.where(x_assign_rounded[i] == 1)[0]
                        if len(assigned_facs) != 1:
                            valid = False
                            break
                        f = assigned_facs[0]
                        load[f] += demands[i]
                    
                    if np.any(load > capacities + 1e-9):
                        valid = False
                        
                    if valid:
                        result = np.zeros(n_cust, dtype=int)
                        for i in range(n_cust):
                            result[i] = np.where(x_assign_rounded[i] == 1)[0][0]
                        milp_solution = result
    except Exception:
        milp_solution = None

    # Choose the best solution
    if milp_solution is not None:
        # Compare with fallback
        cost_milp = compute_cost(milp_solution)
        cost_fallback = compute_cost(best_fallback)
        
        if cost_milp <= cost_fallback:
            return milp_solution
        else:
            return best_fallback
    else:
        return best_fallback
