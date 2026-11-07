"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import sys


def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # We formulate this as a linear programming problem.
    # Decision variables:
    # x[i, j] = 1 if customer i is assigned to facility j, else 0
    # y[j] = 1 if facility j is open, else 0
    
    # Objective: Minimize sum_j open_cost[j] * y[j] + sum_i sum_j assign_cost[i, j] * x[i, j]
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility:
    #    sum_j x[i, j] = 1 for all i
    # 2. Facility capacity:
    #    sum_i demands[i] * x[i, j] <= capacities[j] for all j
    # 3. Linking constraints (if customer assigned to facility, facility must be open):
    #    x[i, j] <= y[j] for all i, j
    #    (This implies y[j] >= x[i, j] for any i. Since y[j] is binary and we minimize cost, 
    #     y[j] will be 1 if any x[i, j] is 1, and 0 otherwise.)
    
    # Total variables: n_customers * n_facilities (for x) + n_facilities (for y)
    n_vars = n_customers * n_facilities + n_facilities
    
    # Objective coefficients
    c = np.zeros(n_vars)
    for i in range(n_customers):
        for j in range(n_facilities):
            c[i * n_facilities + j] = assign_cost[i, j]
    for j in range(n_facilities):
        c[n_customers * n_facilities + j] = open_cost[j]
    
    # Equality constraints: sum_j x[i, j] = 1 for each customer i
    A_eq = np.zeros((n_customers, n_vars))
    b_eq = np.ones(n_customers)
    for i in range(n_customers):
        for j in range(n_facilities):
            A_eq[i, i * n_facilities + j] = 1.0
    
    # Inequality constraints:
    # 1. Capacity: sum_i demands[i] * x[i, j] <= capacities[j]
    # 2. Linking: x[i, j] - y[j] <= 0  =>  x[i, j] + (-1)*y[j] <= 0
    
    n_ineq = n_facilities + n_customers * n_facilities
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    # Capacity constraints
    for j in range(n_facilities):
        for i in range(n_customers):
            A_ub[j, i * n_facilities + j] = demands[i]
        b_ub[j] = capacities[j]
    
    # Linking constraints
    idx = n_facilities
    for i in range(n_customers):
        for j in range(n_facilities):
            A_ub[idx, i * n_facilities + j] = 1.0
            A_ub[idx, n_customers * n_facilities + j] = -1.0
            b_ub[idx] = 0.0
            idx += 1
    
    # Bounds: 0 <= x[i, j] <= 1, 0 <= y[j] <= 1
    bounds = [(0, 1)] * n_vars
    
    # Solve LP relaxation
    # Note: The constraint matrix is totally unimodular for this formulation?
    # Actually, the linking constraints x[i,j] <= y[j] with sum_j x[i,j]=1 and capacities
    # might not guarantee integrality for all instances, but for this specific structure
    # (assignment with fixed charge), the LP relaxation often yields integer solutions
    # or very close ones. However, scipy's linprog solves the LP relaxation.
    # If the solution is not integer, we need to round or use a MILP solver.
    # Since we only have LP, we rely on the fact that for small instances (100x20)
    # and typical data, the LP solution might be integer or we can post-process.
    # However, the problem states "integers from 5 to 35", "capacities integers".
    # Let's try solving the LP first.
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback to greedy if LP fails
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
    
    # Check if solution is sufficiently integer
    # If not, we perform a simple rounding heuristic:
    # Assign each customer to the facility with the highest x value, checking capacity.
    # This is a greedy repair.
    
    # First, check if the LP solution is basically integer
    if np.allclose(x_vals, np.round(x_vals), atol=1e-4):
        x_rounded = np.round(x_vals).astype(int)
        # Verify capacity
        loads = x_rounded.T @ demands
        if np.all(loads <= capacities + 1e-6):
            result = x_rounded.argmax(axis=1)
            return result
    
    # If not integer, use a greedy repair based on LP guidance
    # Sort customers by demand descending (largest first)
    # For each customer, pick the best facility (lowest cost) that has remaining capacity
    # But we can prioritize facilities with high LP values?
    # Let's just do a standard greedy but maybe use LP solution to break ties or guide.
    # Actually, a simple greedy is safer if LP fails integrality.
    
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    # Sort by demand descending
    order = np.argsort(-demands)
    for i in order:
        # Get costs for this customer
        costs = assign_cost[i].copy()
        # Mask full facilities
        for f in range(n_facilities):
            if load[f] + demands[i] > capacities[f]:
                costs[f] = np.inf
        # If all are inf, this is a problem, but instance should be feasible
        # Pick min cost
        min_idx = np.argmin(costs)
        if np.isinf(costs[min_idx]):
            # Try to find any feasible, maybe relax? No, must be feasible.
            # If no feasible, this instance is infeasible? But task says "assigns each...".
            # Assume feasible.
            pass
        result[i] = min_idx
        load[min_idx] += demands[i]
    
    return result
# EVOLVE-BLOCK-END
