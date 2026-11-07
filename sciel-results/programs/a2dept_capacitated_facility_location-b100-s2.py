"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import sys

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Try to use MILP for exact solution
    try:
        # Variables:
        # x[i, j] for i in range(n_customers), j in range(n_facilities)
        # y[j] for j in range(n_facilities)
        # Total variables: n_customers * n_facilities + n_facilities
        
        n_x = n_customers * n_facilities
        n_y = n_facilities
        n_vars = n_x + n_y
        
        # Objective coefficients
        c = np.zeros(n_vars)
        # x costs
        c[:n_x] = assign_cost.flatten()
        # y costs
        c[n_x:] = open_cost
        
        # Integrality: x and y are binary (0 or 1)
        # milp integrality: 0 for continuous, 1 for integer
        # Since x and y are binary, we set integrality to 1 for all, 
        # but we need to ensure bounds are [0,1] to make them binary.
        integrality = np.ones(n_vars)
        
        # Bounds
        # x in [0,1], y in [0,1]
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        bounds = Bounds(lb, ub)
        
        # Constraints:
        # 1. Assignment: sum_j x[i, j] = 1 for each i
        #    This is an equality constraint.
        #    Matrix A_eq: n_customers x n_vars
        #    Row i: 1s in columns corresponding to x[i, 0..M-1]
        
        A_eq = np.zeros((n_customers, n_vars))
        b_eq = np.ones(n_customers)
        for i in range(n_customers):
            for j in range(n_facilities):
                A_eq[i, i * n_facilities + j] = 1.0
        
        # 2. Capacity: sum_i demands[i] * x[i, j] <= capacities[j] for each j
        #    This is an inequality constraint.
        #    Matrix A_ub: n_facilities x n_vars
        #    Row j: demands[i] in columns corresponding to x[i, j]
        
        A_ub = np.zeros((n_facilities, n_vars))
        b_ub = np.array(capacities)
        for j in range(n_facilities):
            for i in range(n_customers):
                A_ub[j, i * n_facilities + j] = demands[i]
        
        # 3. Linking: x[i, j] <= y[j]  =>  x[i, j] - y[j] <= 0
        #    This is an inequality constraint.
        #    Matrix A_link: n_customers * n_facilities x n_vars
        #    Row (i, j): 1 in x[i, j], -1 in y[j]
        
        n_links = n_customers * n_facilities
        A_link = np.zeros((n_links, n_vars))
        b_link = np.zeros(n_links)
        idx = 0
        for i in range(n_customers):
            for j in range(n_facilities):
                A_link[idx, i * n_facilities + j] = 1.0
                A_link[idx, n_x + j] = -1.0
                b_link[idx] = 0.0
                idx += 1
        
        # Combine inequality constraints
        # A_ub_total = [A_ub; A_link]
        # b_ub_total = [b_ub; b_link]
        A_ub_total = np.vstack((A_ub, A_link))
        b_ub_total = np.concatenate((b_ub, b_link))
        
        # Create LinearConstraint
        # LinearConstraint(A, lb, ub)
        # For equality: lb == ub
        # For inequality: lb == -inf, ub == b_ub_total
        # We can combine them or keep separate. 
        # milp accepts a list of LinearConstraints or a single one?
        # It accepts a single LinearConstraint object or a list? 
        # Documentation says: "linear_constraint : LinearConstraint"
        # It seems we can only pass one. We must combine equality and inequality.
        # Equality: sum x = 1  => 1 <= sum x <= 1
        # Inequality: ... <= cap
        
        # Let's build a single matrix for all constraints.
        # Rows 0..n_customers-1: Equality (Assignment)
        # Rows n_customers..n_customers+n_facilities-1: Inequality (Capacity)
        # Rows n_customers+n_facilities..: Inequality (Linking)
        
        n_rows = n_customers + n_facilities + n_links
        A_all = np.zeros((n_rows, n_vars))
        lb_all = np.zeros(n_rows)
        ub_all = np.zeros(n_rows)
        
        # 1. Assignment Equalities
        for i in range(n_customers):
            for j in range(n_facilities):
                A_all[i, i * n_facilities + j] = 1.0
            lb_all[i] = 1.0
            ub_all[i] = 1.0
            
        # 2. Capacity Inequalities
        offset = n_customers
        for j in range(n_facilities):
            for i in range(n_customers):
                A_all[offset + j, i * n_facilities + j] = demands[i]
            lb_all[offset + j] = -np.inf
            ub_all[offset + j] = capacities[j]
            
        # 3. Linking Inequalities
        offset += n_facilities
        idx = 0
        for i in range(n_customers):
            for j in range(n_facilities):
                A_all[offset + idx, i * n_facilities + j] = 1.0
                A_all[offset + idx, n_x + j] = -1.0
                lb_all[offset + idx] = -np.inf
                ub_all[offset + idx] = 0.0
                idx += 1
        
        linear_constraint = LinearConstraint(A_all, lb_all, ub_all)
        
        res = milp(c, integrality=integrality, bounds=bounds, constraints=linear_constraint)
        
        if res.success:
            x_vals = res.x[:n_x].reshape(n_customers, n_facilities)
            # Round to nearest integer (should be 0 or 1)
            x_rounded = np.round(x_vals).astype(int)
            
            # Verify constraints just in case of numerical issues
            # 1. Each customer assigned to exactly one
            if not np.all(x_rounded.sum(axis=1) == 1):
                raise ValueError("Invalid assignment: customer not assigned to exactly one facility")
            # 2. Capacity
            loads = x_rounded.T @ demands
            if np.any(loads > capacities):
                raise ValueError("Invalid assignment: capacity exceeded")
            
            result = x_rounded.argmax(axis=1)
            return result
            
    except Exception:
        pass
    
    # Fallback to LP relaxation + Greedy Repair if MILP fails or is not available
    # This part is similar to the original but kept as fallback
    
    n_vars = n_customers * n_facilities + n_facilities
    c = np.zeros(n_vars)
    c[:n_customers * n_facilities] = assign_cost.flatten()
    c[n_customers * n_facilities:] = open_cost
    
    A_eq = np.zeros((n_customers, n_vars))
    b_eq = np.ones(n_customers)
    for i in range(n_customers):
        for j in range(n_facilities):
            A_eq[i, i * n_facilities + j] = 1.0
            
    n_ineq = n_facilities + n_customers * n_facilities
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    for j in range(n_facilities):
        for i in range(n_customers):
            A_ub[j, i * n_facilities + j] = demands[i]
        b_ub[j] = capacities[j]
        
    idx = n_facilities
    for i in range(n_customers):
        for j in range(n_facilities):
            A_ub[idx, i * n_facilities + j] = 1.0
            A_ub[idx, n_customers * n_facilities + j] = -1.0
            b_ub[idx] = 0.0
            idx += 1
            
    bounds = [(0, 1)] * n_vars
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res.success:
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        return result
        
    x_vals = res.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
    
    if np.allclose(x_vals, np.round(x_vals), atol=1e-4):
        x_rounded = np.round(x_vals).astype(int)
        loads = x_rounded.T @ demands
        if np.all(loads <= capacities + 1e-6):
            result = x_rounded.argmax(axis=1)
            return result
            
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    order = np.argsort(-demands)
    for i in order:
        costs = assign_cost[i].copy()
        for f in range(n_facilities):
            if load[f] + demands[i] > capacities[f]:
                costs[f] = np.inf
        min_idx = np.argmin(costs)
        if np.isinf(costs[min_idx]):
            pass
        result[i] = min_idx
        load[min_idx] += demands[i]
        
    return result
# EVOLVE-BLOCK-END
