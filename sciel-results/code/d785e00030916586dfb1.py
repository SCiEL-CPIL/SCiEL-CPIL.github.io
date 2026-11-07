"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    """
    Solve the capacitated facility location problem using MILP (HiGHS via SciPy).
    
    Parameters:
    -----------
    demands : array-like, shape (100,)
        Integer demand per customer.
    capacities : array-like, shape (20,)
        Integer capacity per facility.
    open_cost : array-like, shape (20,)
        Fixed cost to open each facility.
    assign_cost : array-like, shape (100, 20)
        Cost to assign customer i to facility j.
    
    Returns:
    --------
    result : numpy.ndarray, shape (100,)
        Facility index assigned to each customer.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin within 120s limit
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Decision variables:
    # x[i, j] = 1 if customer i is assigned to facility j
    # y[j] = 1 if facility j is open
    # Total variables: n_customers * n_facilities + n_facilities
    
    # Objective: minimize sum(open_cost[j] * y[j]) + sum(assign_cost[i,j] * x[i,j])
    c = np.concatenate([open_cost, assign_cost.ravel()])
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility: sum_j x[i,j] = 1 for all i
    # 2. Capacity: sum_i demands[i] * x[i,j] <= capacities[j] * y[j] for all j
    #    Rearranged: sum_i demands[i] * x[i,j] - capacities[j] * y[j] <= 0
    
    # Build constraint matrix
    # Variables order: [y[0], y[1], ..., y[n_f-1], x[0,0], x[0,1], ..., x[n_c-1, n_f-1]]
    
    # Constraint 1: assignment (n_customers constraints)
    A_assign = np.zeros((n_customers, n_customers * n_facilities + n_facilities))
    for i in range(n_customers):
        for j in range(n_facilities):
            A_assign[i, n_facilities + i * n_facilities + j] = 1.0
    
    # Constraint 2: capacity (n_facilities constraints)
    A_cap = np.zeros((n_facilities, n_customers * n_facilities + n_facilities))
    for j in range(n_facilities):
        # -capacities[j] * y[j]
        A_cap[j, j] = -capacities[j]
        # + demands[i] * x[i,j]
        for i in range(n_customers):
            A_cap[j, n_facilities + i * n_facilities + j] = demands[i]
    
    # Combine constraints
    A = np.vstack([A_assign, A_cap])
    
    # Bounds:
    # y[j] in {0, 1} -> integrality 1, bounds [0, 1]
    # x[i,j] in {0, 1} -> integrality 1, bounds [0, 1]
    integrality = np.ones(n_customers * n_facilities + n_facilities)
    lb = np.zeros(n_customers * n_facilities + n_facilities)
    ub = np.ones(n_customers * n_facilities + n_facilities)
    
    # Constraint bounds:
    # Assignment: equality, so lb = 1, ub = 1
    # Capacity: <= 0, so lb = -inf, ub = 0
    cons_lb = np.concatenate([np.ones(n_customers), np.full(n_facilities, -np.inf)])
    cons_ub = np.concatenate([np.ones(n_customers), np.zeros(n_facilities)])
    
    constraints = LinearConstraint(A, cons_lb, cons_ub)
    bounds = Bounds(lb, ub)
    
    # Solve MILP
    result = None
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={"time_limit": 100})
        
        if res.status == 0:  # Optimal
            x = res.x
            y = x[:n_facilities]
            x_assign = x[n_facilities:].reshape(n_customers, n_facilities)
            
            # Determine assignment from x_assign
            result = np.zeros(n_customers, dtype=int)
            for i in range(n_customers):
                # Find the facility with x[i,j] closest to 1
                j = int(np.argmax(x_assign[i]))
                result[i] = j
        elif res.status == 2:  # Locally optimal
            x = res.x
            y = x[:n_facilities]
            x_assign = x[n_facilities:].reshape(n_customers, n_facilities)
            
            result = np.zeros(n_customers, dtype=int)
            for i in range(n_customers):
                j = int(np.argmax(x_assign[i]))
                result[i] = j
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        # Fall back to greedy if MILP took too long or failed
        if result is None:
            result = _greedy_fallback(demands, capacities, open_cost, assign_cost)
        return result
    
    # Verify and fix if needed
    if result is None:
        result = _greedy_fallback(demands, capacities, open_cost, assign_cost)
    else:
        result = _verify_and_fix(result, demands, capacities, open_cost, assign_cost)
    
    return result


def _greedy_fallback(demands, capacities, open_cost, assign_cost):
    """Greedy fallback: largest demand first, cheapest feasible facility."""
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # No feasible facility found, assign to least loaded
            f = int(np.argmin(load))
            load[f] += demands[i]
            result[i] = f
    
    return result


def _verify_and_fix(result, demands, capacities, open_cost, assign_cost):
    """Verify the assignment is valid; if not, use greedy fallback."""
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Check indices
    if np.any(result < 0) or np.any(result >= n_facilities):
        return _greedy_fallback(demands, capacities, open_cost, assign_cost)
    
    # Check capacities
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands[i]
    
    if np.any(load > capacities + 1e-9):
        return _greedy_fallback(demands, capacities, open_cost, assign_cost)
    
    return result
# EVOLVE-BLOCK-END
