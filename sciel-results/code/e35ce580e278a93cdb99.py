import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds, linprog

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Lagrangian Relaxation for a strong initial guess
    # This provides a global perspective on capacity constraints, unlike pure greedy.
    # It guides the initial assignment to be closer to the optimal structure.
    def lagrangian_init(demands, capacities, open_cost, assign_cost):
        n_c = len(demands)
        n_f = len(capacities)
        
        # Initialize dual variables (prices for capacity constraints)
        # Start with 0 or small values.
        mu = np.zeros(n_f)
        
        # Subgradient optimization
        max_iter = 100
        best_assign = None
        best_load = None
        
        # Simple subgradient method
        # Step size parameter
        step_param = 1.0
        
        for k in range(max_iter):
            # 1. Solve Lagrangian subproblem
            # Minimize sum_i sum_f (c_if + mu_f) x_if
            # s.t. sum_f x_if = 1 for all i
            # x_if >= 0
            
            # For each customer i, pick f that minimizes c_if + mu_f
            costs = assign_cost + mu[None, :]
            best_f = np.argmin(costs, axis=1)
            
            # 2. Check capacity constraints
            load = np.zeros(n_f)
            for i in range(n_c):
                load[best_f[i]] += demands[i]
            
            # 3. Check optimality / convergence
            # If all capacities are satisfied, this is a feasible solution to the original problem.
            # Note: It might not be optimal, but it's a good candidate.
            if np.all(load <= capacities + 1e-6):
                if best_assign is None or True: # We found a feasible one
                    # We can stop early or continue to find better mu? 
                    # Usually, if feasible, we can use it.
                    # But we might find a better one. Let's collect feasible solutions.
                    if best_assign is None:
                        best_assign = best_f.copy()
                        best_load = load.copy()
            
            # 4. Update mu (Subgradient)
            # Gradient g_f = load_f - capacity_f
            grad = load - capacities
            
            # Norm of gradient
            g_norm = np.linalg.norm(grad)
            if g_norm < 1e-6:
                break
            
            # Update rule: mu_f = max(0, mu_f - step * g_f)
            # Step size: step_param / g_norm or decaying
            step = step_param / g_norm
            mu = np.maximum(0, mu - step * grad)
            
            # Decay step_param slightly?
            step_param *= 0.95
            
        # If no feasible solution was found during the iterations (which is possible if the 
        # Lagrangian relaxation doesn't naturally satisfy capacities), 
        # we need a fallback feasible assignment.
        if best_assign is None:
            # Fallback: Greedy assignment
            # Sort customers by demand descending
            order = np.argsort(-demands)
            result = np.full(n_c, -1, dtype=int)
            load = np.zeros(n_f)
            
            for i in order:
                # Find cheapest feasible facility
                fac_order = np.argsort(assign_cost[i])
                for f in fac_order:
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        break
                else:
                    # Should not happen if total capacity >= total demand
                    # Assign to least loaded feasible? Or raise error?
                    # Task implies feasible instances.
                    # For robustness, pick any facility with space
                    for f in range(n_f):
                        if load[f] + demands[i] <= capacities[f]:
                            load[f] += demands[i]
                            result[i] = f
                            break
            best_assign = result
            best_load = load
            
        return best_assign

    # 2. MIP Formulation
    # Variables:
    # x[i, f] in {0, 1} : 1 if customer i is assigned to facility f
    # y[f] in {0, 1} : 1 if facility f is open
    
    # Objective:
    # Minimize sum_{i,f} assign_cost[i,f] * x[i,f] + sum_f open_cost[f] * y[f]
    
    # Constraints:
    # 1. sum_f x[i,f] = 1 for all i (Each customer assigned to exactly one facility)
    # 2. sum_i demands[i] * x[i,f] <= capacities[f] * y[f] for all f (Capacity constraint)
    #    Note: This implies y[f] must be 1 if any customer is assigned to f.
    #    If y[f]=0, then sum_i demands[i] x[i,f] <= 0, which means x[i,f]=0 for all i.
    #    So this single constraint handles both capacity and open/close logic.
    
    # Flatten x and y into a single vector for scipy.optimize.milp
    # Order: x[0,0], x[0,1], ..., x[0,F-1], x[1,0], ..., x[n_c-1, F-1], y[0], ..., y[F-1]
    
    n_vars = n_customers * n_facilities + n_facilities
    
    # Objective coefficients
    c = np.zeros(n_vars)
    for i in range(n_customers):
        for f in range(n_facilities):
            c[i * n_facilities + f] = assign_cost[i, f]
    for f in range(n_facilities):
        c[n_customers * n_facilities + f] = open_cost[f]
    
    # Bounds
    # All variables are binary, so bounds are [0, 1]
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    bounds = Bounds(lb, ub)
    
    # Integrality
    integrality = np.ones(n_vars)
    
    # Constraints Matrix A
    # We need to build A such that lb_constr <= A @ vars <= ub_constr
    
    # Constraint 1: sum_f x[i,f] = 1 for each i
    # There are n_customers such constraints.
    # Row for customer i:
    # x[i,0] + ... + x[i, F-1] = 1
    # In the flattened vector, x[i,f] is at index i * n_facilities + f.
    
    # Constraint 2: sum_i demands[i] x[i,f] <= capacities[f] * y[f] for each f
    # Rearranged: sum_i demands[i] x[i,f] - capacities[f] * y[f] <= 0
    # There are n_facilities such constraints.
    # Row for facility f:
    # sum_i demands[i] * x[i,f] - capacities[f] * y[f] <= 0
    # Indices:
    # x[i,f] is at i * n_facilities + f.
    # y[f] is at n_customers * n_facilities + f.
    
    # Total rows: n_customers + n_facilities
    n_rows = n_customers + n_facilities
    A = np.zeros((n_rows, n_vars))
    lb_constr = np.zeros(n_rows)
    ub_constr = np.zeros(n_rows)
    
    # Fill Constraint 1 (Customer assignment)
    for i in range(n_customers):
        row_idx = i
        for f in range(n_facilities):
            col_idx = i * n_facilities + f
            A[row_idx, col_idx] = 1.0
        lb_constr[row_idx] = 1.0
        ub_constr[row_idx] = 1.0
        
    # Fill Constraint 2 (Capacity)
    for f in range(n_facilities):
        row_idx = n_customers + f
        for i in range(n_customers):
            col_idx = i * n_facilities + f
            A[row_idx, col_idx] = demands[i]
        # Coefficient for y[f]
        col_idx_y = n_customers * n_facilities + f
        A[row_idx, col_idx_y] = -capacities[f]
        lb_constr[row_idx] = -np.inf
        ub_constr[row_idx] = 0.0
        
    constraints = LinearConstraint(A, lb_constr, ub_constr)
    
    # Solve MIP
    # Use the Lagrangian result as a warm start? 
    # scipy.optimize.milp does not directly accept a warm start solution in the same way,
    # but we can check if the Lagrangian solution is good enough or just let HiGHS do its thing.
    # Given the small size (100x20 = 2000 vars + 20 = 2020 vars, 120 constraints),
    # HiGHS should be fast.
    
    start_time = time.time()
    deadline = 140.0
    
    try:
        res = milp(c=c,
                   constraints=constraints,
                   integrality=integrality,
                   bounds=bounds,
                   options={'time_limit': deadline - start_time - 5.0, 'mip_rel_gap': 0.001})
        
        if res.status == 0: # Optimal or feasible
            x_flat = res.x
            # Extract x
            x = x_flat[:n_customers * n_facilities].reshape(n_customers, n_facilities)
            
            # Assign each customer to the facility with x[i,f] closest to 1
            # Since it's a MIP, x should be 0 or 1.
            # If it's relaxed (LP relaxation), we might need rounding.
            # But milp with integrality=1 returns integer solutions.
            
            assignment = np.argmax(x, axis=1)
            
            # Verify feasibility
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[assignment[i]] += demands[i]
            
            if np.all(load <= capacities + 1e-6):
                return assignment
            else:
                # If MIP returns infeasible or invalid, fall back to Lagrangian/Greedy
                pass
                
    except Exception:
        # If MIP fails, fall back
        pass
        
    # Fallback: Use Lagrangian initialized solution
    return lagrangian_init(demands, capacities, open_cost, assign_cost)
