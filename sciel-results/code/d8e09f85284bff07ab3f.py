"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_objective(assignment, demands, capacities, open_cost, assign_cost):
    """Compute the total cost of an assignment."""
    n_cust = len(demands)
    n_fac = len(capacities)
    # Check validity
    load = np.zeros(n_fac)
    for i in range(n_cust):
        f = assignment[i]
        if f < 0 or f >= n_fac:
            return np.inf
        load[f] += demands[i]
    for f in range(n_fac):
        if load[f] > capacities[f] + 1e-9:
            return np.inf
    # Compute cost
    total = 0.0
    used = np.zeros(n_fac, dtype=bool)
    for i in range(n_cust):
        f = assignment[i]
        total += assign_cost[i, f]
        used[f] = True
    total += np.sum(open_cost[used])
    return total


def _greedy_init(demands, capacities, open_cost, assign_cost, order=None):
    """Greedy initialization with given order (default: largest demand first)."""
    n_cust = len(demands)
    n_fac = len(capacities)
    if order is None:
        order = np.argsort(-demands)
    else:
        order = np.array(order)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    for i in order:
        best_f = -1
        best_cost = np.inf
        for f in range(n_fac):
            if load[f] + demands[i] <= capacities[f]:
                c = assign_cost[i, f]
                if c < best_cost:
                    best_cost = c
                    best_f = f
        if best_f >= 0:
            load[best_f] += demands[i]
            result[i] = best_f
        else:
            # Fallback: assign to facility with most remaining capacity
            rem = capacities - load
            best_f = int(np.argmax(rem))
            load[best_f] += demands[i]
            result[i] = best_f
    return result


def _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline):
    """Local search with reassignment moves."""
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[assignment[i]] += demands[i]
    
    # Count customers per facility
    cust_count = np.zeros(n_fac, dtype=int)
    for i in range(n_cust):
        cust_count[assignment[i]] += 1
    
    current_cost = _compute_objective(assignment, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_cost):
        return assignment, current_cost
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try reassigning each customer
        for i in range(n_cust):
            if time.time() >= deadline:
                break
            f_old = assignment[i]
            for f_new in range(n_fac):
                if f_new == f_old:
                    continue
                # Capacity check
                if load[f_new] + demands[i] > capacities[f_new]:
                    continue
                # Compute delta cost
                delta = assign_cost[i, f_new] - assign_cost[i, f_old]
                # Opening cost changes
                if cust_count[f_new] == 0:
                    delta += open_cost[f_new]  # New facility opened
                if cust_count[f_old] == 1:
                    delta -= open_cost[f_old]  # Old facility closed
                if delta < -1e-9:
                    # Make the move
                    load[f_old] -= demands[i]
                    load[f_new] += demands[i]
                    cust_count[f_old] -= 1
                    cust_count[f_new] += 1
                    assignment[i] = f_new
                    current_cost += delta
                    improved = True
                    break
    return assignment, current_cost


def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Multiple restarts with different orderings
    best_assignment = None
    best_cost = np.inf
    
    # First: standard greedy (largest demand first)
    t0 = time.time()
    assignment = _greedy_init(demands, capacities, open_cost, assign_cost)
    assignment, cost = _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline)
    if cost < best_cost:
        best_cost = cost
        best_assignment = assignment.copy()
    
    # Additional restarts with random orderings
    rng = np.random.RandomState(42)
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() >= deadline:
            break
        # Random order
        order = rng.permutation(n_cust)
        assignment = _greedy_init(demands, capacities, open_cost, assign_cost, order=order)
        assignment, cost = _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline)
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment.copy()
    
    # Final check
    if best_assignment is None:
        # Fallback: all to facility 0 (may be invalid but better than nothing)
        best_assignment = np.zeros(n_cust, dtype=int)
    
    # Validate and return
    return best_assignment
# EVOLVE-BLOCK-END
