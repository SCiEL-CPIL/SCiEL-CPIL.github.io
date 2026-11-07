"""Facility location assignment using MIP via scipy.optimize.milp with deadline guard."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def _solve_mip(demands, capacities, open_cost, assign_cost, time_limit=100):
    """
    Solve the capacitated facility location problem as a MILP.
    
    Variables:
      x[i, f] in {0,1}: customer i assigned to facility f
      y[f] in {0,1}: facility f is open
    
    Objective:
      minimize sum_f open_cost[f] * y[f] + sum_i,f assign_cost[i,f] * x[i,f]
    
    Constraints:
      For each customer i: sum_f x[i,f] = 1
      For each facility f: sum_i demands[i] * x[i,f] <= capacities[f] * y[f]
      y[f] >= x[i,f] for all i,f (implied by capacity if capacities>0, but we add explicitly)
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Number of binary variables: n_customers * n_facilities (x) + n_facilities (y)
    n_x = n_customers * n_facilities
    n_y = n_facilities
    n_vars = n_x + n_y
    
    # Objective coefficients
    c = np.zeros(n_vars)
    # x variables
    for i in range(n_customers):
        for f in range(n_facilities):
            c[i * n_facilities + f] = assign_cost[i, f]
    # y variables
    for f in range(n_facilities):
        c[n_x + f] = open_cost[f]
    
    # Bounds: all binary
    integrality = np.ones(n_vars)
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    bounds = Bounds(lb, ub)
    
    # Constraints in A @ var <= b form (or LinearConstraint)
    # We'll use LinearConstraint with (lb, ub)
    
    constraint_matrix_rows = []
    constraint_lb = []
    constraint_ub = []
    
    # 1) Each customer assigned to exactly one facility: sum_f x[i,f] = 1
    for i in range(n_customers):
        row = np.zeros(n_vars)
        for f in range(n_facilities):
            row[i * n_facilities + f] = 1.0
        constraint_matrix_rows.append(row)
        constraint_lb.append(1.0)
        constraint_ub.append(1.0)
    
    # 2) Capacity constraints: sum_i demands[i] * x[i,f] <= capacities[f] * y[f]
    #    => sum_i demands[i] * x[i,f] - capacities[f] * y[f] <= 0
    for f in range(n_facilities):
        row = np.zeros(n_vars)
        for i in range(n_customers):
            row[i * n_facilities + f] = demands[i]
        row[n_x + f] = -capacities[f]
        constraint_matrix_rows.append(row)
        constraint_lb.append(-np.inf)
        constraint_ub.append(0.0)
    
    # 3) y[f] >= x[i,f] for all i,f
    #    => x[i,f] - y[f] <= 0
    for i in range(n_customers):
        for f in range(n_facilities):
            row = np.zeros(n_vars)
            row[i * n_facilities + f] = 1.0
            row[n_x + f] = -1.0
            constraint_matrix_rows.append(row)
            constraint_lb.append(-np.inf)
            constraint_ub.append(0.0)
    
    A = np.array(constraint_matrix_rows)
    lb_con = np.array(constraint_lb)
    ub_con = np.array(constraint_ub)
    
    constraints = LinearConstraint(A, lb_con, ub_con)
    
    start_time = time.time()
    effective_time_limit = max(1.0, time_limit - (time.time() - start_time) - 5.0)
    
    try:
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": effective_time_limit, "presolve": True}
        )
    except Exception:
        res = None
    
    if res is not None and res.x is not None:
        # Extract assignment from x variables
        x_vals = res.x[:n_x].reshape(n_customers, n_facilities)
        # Each customer should have exactly one 1; if not (due to numerical issues), pick argmin
        assignment = np.zeros(n_customers, dtype=int)
        for i in range(n_customers):
            row = x_vals[i]
            # If the row sums to approximately 1, pick the argmax
            if np.sum(row) > 0.5:
                assignment[i] = int(np.argmax(row))
            else:
                # Fallback: cheapest facility
                assignment[i] = int(np.argmin(assign_cost[i]))
        
        # Validate: check capacities
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        
        if np.all(load <= capacities + 1e-6):
            return assignment
    
    # Fallback: greedy heuristic
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    order = np.argsort(-demands)
    for i in order:
        feasible = []
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                feasible.append((assign_cost[i, f], f))
        if feasible:
            feasible.sort()
            f = feasible[0][1]
            load[f] += demands[i]
            result[i] = f
        else:
            # Should not happen, but pick any
            result[i] = 0
    return result


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    
    Parameters:
        demands: array of 100 integers (5-35)
        capacities: array of 20 integers
        open_cost: array of 20 floats (50-150)
        assign_cost: array of shape (100, 20)
    
    Returns:
        integer array of 100 facility indices
    """
    demands = np.asarray(demands, dtype=int)
    capacities = np.asarray(capacities, dtype=int)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Deadline guard: total budget is 150s, we want to return well before
    start_time = time.time()
    
    # Try MIP with a time limit that leaves margin
    time_limit_mip = 100.0
    result = _solve_mip(demands, capacities, open_cost, assign_cost, time_limit=time_limit_mip)
    
    elapsed = time.time() - start_time
    
    # If we're getting close to the deadline, just return what we have
    if elapsed > 140.0:
        return result
    
    return result
