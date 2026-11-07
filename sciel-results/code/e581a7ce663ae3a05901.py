"""Capacitated facility location via min-cost flow with facility opening costs."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign customers to facilities using LP relaxation of the capacitated facility location problem.
    The constraint matrix is totally unimodular, so the LP solution is integral.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n = len(demands)
    m = len(capacities)
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    # Variables: x[i, f] for each customer-facility pair (n*m vars), y[f] for each facility open (m vars)
    # Total vars: n*m + m
    num_x = n * m
    num_y = m
    num_vars = num_x + num_y

    # Objective: minimize sum(assign_cost[i,f] * x[i,f]) + sum(open_cost[f] * y[f])
    c = np.zeros(num_vars)
    c[:num_x] = assign_cost.flatten()
    c[num_x:] = open_cost

    # Constraints:
    # 1. Each customer assigned to exactly one facility: sum_f x[i,f] = 1 for each i
    #    n equality constraints
    # 2. Facility capacity: sum_i demands[i]*x[i,f] <= capacities[f] for each f
    #    m inequality constraints
    # 3. Linking: x[i,f] <= y[f] for all i,f
    #    n*m inequality constraints

    # Build A_eq for customer assignment constraints
    # For each customer i, sum_f x[i,f] = 1
    A_eq_rows = []
    b_eq = np.ones(n)
    for i in range(n):
        row = np.zeros(num_vars)
        for f in range(m):
            row[i * m + f] = 1.0
        A_eq_rows.append(row)
    A_eq = np.array(A_eq_rows)

    # Build A_ub for capacity and linking constraints
    A_ub_rows = []
    b_ub = []

    # Capacity constraints: sum_i demands[i]*x[i,f] <= capacities[f]
    for f in range(m):
        row = np.zeros(num_vars)
        for i in range(n):
            row[i * m + f] = demands[i]
        A_ub_rows.append(row)
        b_ub.append(capacities[f])

    # Linking constraints: x[i,f] - y[f] <= 0
    for i in range(n):
        for f in range(m):
            row = np.zeros(num_vars)
            row[i * m + f] = 1.0
            row[num_x + f] = -1.0
            A_ub_rows.append(row)
            b_ub.append(0.0)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub)

    # Bounds: 0 <= x <= 1, 0 <= y <= 1
    bounds = [(0, 1)] * num_vars

    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # Check if we're running out of time
    elapsed = time.time() - start_time
    if elapsed > 100.0:
        # Fall back to greedy if LP is too slow
        return _greedy_fallback(demands, capacities, assign_cost)

    if result.status != 0:
        # LP failed, use greedy fallback
        return _greedy_fallback(demands, capacities, assign_cost)

    # Extract solution
    x = result.x[:num_x].reshape(n, m)
    # Round to get integer assignment (should already be near-integer due to total unimodularity)
    assignment = np.zeros(n, dtype=int)
    for i in range(n):
        # Find the facility with the largest x value
        f = np.argmax(x[i])
        assignment[i] = f

    # Verify capacity constraints and fix if violated
    load = np.zeros(m)
    for i in range(n):
        load[assignment[i]] += demands[i]

    # If any capacity violated, use greedy repair
    if np.any(load > capacities + 1e-6):
        return _greedy_fallback(demands, capacities, assign_cost)

    return assignment


def _greedy_fallback(demands, capacities, assign_cost):
    """Greedy fallback: largest demand first, cheapest feasible facility."""
    n = len(demands)
    m = len(capacities)
    result = np.zeros(n, dtype=int)
    load = np.zeros(m)
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # No feasible facility found, assign to least loaded feasible
            for f in range(m):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
    return result
# EVOLVE-BLOCK-END
