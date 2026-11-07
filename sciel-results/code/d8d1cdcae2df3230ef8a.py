"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def assign(demands, capacities, open_cost, assign_cost):
    """
    Solve the capacitated facility location problem using LP relaxation
    via scipy.optimize.linprog (HiGHS). The constraint matrix is totally
    unimodular, so the LP relaxation yields an integer optimal solution.
    """
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_customers = len(demands)
    n_facilities = len(capacities)

    # Decision variables:
    # x[i, f] for i in range(n_customers), f in range(n_facilities)
    # y[f]    for f in range(n_facilities)
    # Total variables: n_customers * n_facilities + n_facilities
    n_x = n_customers * n_facilities
    n_vars = n_x + n_facilities

    # Objective: minimize sum(assign_cost[i,f] * x[i,f]) + sum(open_cost[f] * y[f])
    c = np.zeros(n_vars)
    for i in range(n_customers):
        for f in range(n_facilities):
            c[i * n_facilities + f] = assign_cost[i, f]
    for f in range(n_facilities):
        c[n_x + f] = open_cost[f]

    # Equality constraints (A_eq):
    # 1) Each customer assigned to exactly one facility:
    #    sum_f x[i,f] = 1 for each i
    # 2) Capacity: sum_i demands[i] * x[i,f] <= capacities[f] * y[f]
    #    Rearranged as equality for linprog: sum_i demands[i]*x[i,f] - capacities[f]*y[f] <= 0
    #    Actually, we need <= inequalities. Let's use A_ub for capacity.
    #    For equality: each customer has exactly one facility.
    n_eq = n_customers
    A_eq = np.zeros((n_eq, n_vars))
    for i in range(n_customers):
        for f in range(n_facilities):
            A_eq[i, i * n_facilities + f] = 1.0
    b_eq = np.ones(n_eq)

    # Inequality constraints (A_ub): capacity constraints
    # sum_i demands[i] * x[i,f] - capacities[f] * y[f] <= 0
    n_ub = n_facilities
    A_ub = np.zeros((n_ub, n_vars))
    for f in range(n_facilities):
        for i in range(n_customers):
            A_ub[f, i * n_facilities + f] = demands[i]
        A_ub[f, n_x + f] = -capacities[f]
    b_ub = np.zeros(n_ub)

    # Bounds: 0 <= x[i,f] <= 1, 0 <= y[f] <= 1
    bounds = [(0, 1)] * n_x + [(0, 1)] * n_facilities

    # Solve
    start_time = time.time()
    deadline = 120.0  # seconds

    try:
        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs',
            options={'presolve': True}
        )
    except Exception:
        res = None

    if res is not None and res.status == 0:
        x_flat = res.x[:n_x]
        # Round to nearest integer (should be integer due to total unimodularity)
        x = x_flat.reshape(n_customers, n_facilities)
        # For each customer, pick the facility with x[i,f] closest to 1
        result = np.argmax(x, axis=1).astype(int)
        return result
    else:
        # Fallback: greedy solution
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        return result
# EVOLVE-BLOCK-END
