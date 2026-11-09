"""Capacitated facility location solver: MILP with time limit + multi-restart local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DEADLINE_MARGIN = 15.0  # seconds of safety margin before the 150s limit


def _solve_milp(demands, capacities, open_cost, assign_cost, time_limit=60):
    """Solve the capacitated facility location problem via MILP."""
    n_cust = len(demands)
    n_fac = len(capacities)

    n_vars = n_cust * n_fac + n_fac

    # Objective
    c_obj = np.zeros(n_vars)
    for c in range(n_cust):
        for f in range(n_fac):
            c_obj[c * n_fac + f] = assign_cost[c, f]
    for f in range(n_fac):
        c_obj[n_cust * n_fac + f] = open_cost[f]

    # Build constraint matrix efficiently
    A_rows = []
    lb = []
    ub = []

    # Constraint 1: assignment (n_cust rows)
    for c in range(n_cust):
        row = np.zeros(n_vars)
        for f in range(n_fac):
            row[c * n_fac + f] = 1.0
        A_rows.append(row)
        lb.append(1.0)
        ub.append(1.0)

    # Constraint 2: capacity (n_fac rows)
    for f in range(n_fac):
        row = np.zeros(n_vars)
        for c in range(n_cust):
            row[c * n_fac + f] = float(demands[c])
        A_rows.append(row)
        lb.append(0.0)
        ub.append(float(capacities[f]))

    # Constraint 3: linking (n_cust * n_fac rows)
    for c in range(n_cust):
        for f in range(n_fac):
            row = np.zeros(n_vars)
            row[c * n_fac + f] = 1.0
            row[n_cust * n_fac + f] = -1.0
            A_rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)

    A_mat = np.array(A_rows)

    integrality = np.ones(n_vars)
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    constraint = LinearConstraint(A_mat, lb, ub)

    try:
        res = milp(
            c=c_obj,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": time_limit, "mip_rel_gap": 0.01}
        )
    except Exception:
        return None

    if res.x is None:
        return None

    x_vals = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
    result = np.argmax(x_vals, axis=1)

    # Verify feasibility
    load = np.zeros(n_fac)
    for c in range(n_cust):
        load[result[c]] += demands[c]
    if np.any(load > capacities + 1e-6):
        return None
    if np.any(result < 0) or np.any(result >= n_fac):
        return None
    if np.any(np.isnan(result)):
        return None

    return result.astype(int)


def _greedy_assign(demands, capacities, open_cost, assign_cost, seed=0):
    """Greedy fallback: largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)

    order = np.argsort(-demands)
    for c in order:
        best_f = -1
        best_cost = np.inf
        for f in range(n_fac):
            if load[f] + demands[c] <= capacities[f]:
                marginal = assign_cost[c, f]
                if load[f] == 0:
                    marginal += open_cost[f]
                if marginal < best_cost:
                    best_cost = marginal
                    best_f = f
        if best_f == -1:
            best_f = 0
        load[best_f] += demands[c]
        result[c] = best_f

    return result


def _random_feasible(demands, capacities, open_cost, assign_cost, rng):
    """Generate a random feasible assignment."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)

    # Shuffle customers
    order = np.arange(n_cust)
    rng.shuffle(order)

    for c in order:
        feasible = [f for f in range(n_fac) if load[f] + demands[c] <= capacities[f]]
        if not feasible:
            # Assign to facility with most remaining capacity
            best_f = np.argmax(capacities - load)
            result[c] = best_f
            load[best_f] += demands[c]
        else:
            # Weight by inverse cost
            costs = np.array([assign_cost[c, f] for f in feasible])
            # Add small epsilon to avoid zero
            weights = 1.0 / (costs + 1.0)
            weights /= weights.sum()
            f = rng.choice(feasible, p=weights)
            result[c] = f
            load[f] += demands[c]

    return result


def _local_search(demands, capacities, open_cost, assign_cost, result, max_iters=200):
    """Local search: try reassigning customers to reduce cost."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy().astype(int)
    load = np.zeros(n_fac)
    for c in range(n_cust):
        load[result[c]] += demands[c]

    def compute_cost():
        cost = 0.0
        for c in range(n_cust):
            cost += assign_cost[c, result[c]]
        for f in range(n_fac):
            if load[f] > 0:
                cost += open_cost[f]
        return cost

    best_cost = compute_cost()
    no_improve = 0

    for _ in range(max_iters):
        improved = False
        for c in range(n_cust):
            cur_f = result[c]
            for new_f in range(n_fac):
                if new_f == cur_f:
                    continue
                if load[new_f] + demands[c] > capacities[new_f]:
                    continue
                # Compute cost delta
                delta = assign_cost[c, new_f] - assign_cost[c, cur_f]
                # Open cost changes
                if load[cur_f] == demands[c]:
                    # Closing cur_f
                    delta -= open_cost[cur_f]
                if load[new_f] == 0:
                    # Opening new_f
                    delta += open_cost[new_f]

                if delta < -1e-9:
                    # Apply move
                    load[cur_f] -= demands[c]
                    load[new_f] += demands[c]
                    result[c] = new_f
                    best_cost += delta
                    improved = True
                    break
        if not improved:
            break

    return result


def _swap_local_search(demands, capacities, open_cost, assign_cost, result, max_iters=100):
    """Local search with 2-swap moves."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy().astype(int)
    load = np.zeros(n_fac)
    for c in range(n_cust):
        load[result[c]] += demands[c]

    def compute_cost():
        cost = 0.0
        for c in range(n_cust):
            cost += assign_cost[c, result[c]]
        for f in range(n_fac):
            if load[f] > 0:
                cost += open_cost[f]
        return cost

    best_cost = compute_cost()

    for _ in range(max_iters):
        improved = False
        # Try single moves first
        for c in range(n_cust):
            cur_f = result[c]
            for new_f in range(n_fac):
                if new_f == cur_f:
                    continue
                if load[new_f] + demands[c] > capacities[new_f]:
                    continue
                delta = assign_cost[c, new_f] - assign_cost[c, cur_f]
                if load[cur_f] == demands[c]:
                    delta -= open_cost[cur_f]
                if load[new_f] == 0:
                    delta += open_cost[new_f]

                if delta < -1e-9:
                    load[cur_f] -= demands[c]
                    load[new_f] += demands[c]
                    result[c] = new_f
                    best_cost += delta
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # Try 2-swap moves
            for c1 in range(n_cust):
                for c2 in range(c1 + 1, n_cust):
                    f1 = result[c1]
                    f2 = result[c2]
                    if f1 == f2:
                        continue
                    # Swap: c1 -> f2, c2 -> f1
                    if load[f2] - demands[c2] + demands[c1] > capacities[f2]:
                        continue
                    if load[f1] - demands[c1] + demands[c2] > capacities[f1]:
                        continue

                    delta = (assign_cost[c1, f2] - assign_cost[c1, f1]) + \
                            (assign_cost[c2, f1] - assign_cost[c2, f2])

                    # Open cost changes
                    old_load_f1 = load[f1]
                    old_load_f2 = load[f2]
                    new_load_f1 = old_load_f1 - demands[c1] + demands[c2]
                    new_load_f2 = old_load_f2 - demands[c2] + demands[c1]

                    # Facility f1 closes if old_load_f1 == demands[c1] and new_load_f1 == 0
                    # But we're swapping, so f1 still has c2 if demands[c2] > 0
                    # Actually, we need to check if the facility was only serving c1
                    if old_load_f1 == demands[c1] and new_load_f1 == 0:
                        delta -= open_cost[f1]
                    if old_load_f1 == 0 and new_load_f1 > 0:
                        delta += open_cost[f1]

                    if old_load_f2 == demands[c2] and new_load_f2 == 0:
                        delta -= open_cost[f2]
                    if old_load_f2 == 0 and new_load_f2 > 0:
                        delta += open_cost[f2]

                    if delta < -1e-9:
                        load[f1] = new_load_f1
                        load[f2] = new_load_f2
                        result[c1] = f2
                        result[c2] = f1
                        best_cost += delta
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            break

    return result


def assign(demands, capacities, open_cost, assign_cost):
    """Assign each customer to a facility, minimizing total cost."""
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    # Get a feasible solution via greedy
    result = _greedy_assign(demands, capacities, open_cost, assign_cost)

    def compute_cost(res):
        load = np.zeros(n_fac)
        for c in range(n_cust):
            load[res[c]] += demands[c]
        cost = 0.0
        for c in range(n_cust):
            cost += assign_cost[c, res[c]]
        for f in range(n_fac):
            if load[f] > 0:
                cost += open_cost[f]
        return cost

    best_result = result.copy()
    best_cost = compute_cost(result)

    # Try MILP if we have time
    if time.time() < deadline - 40:
        milp_result = _solve_milp(demands, capacities, open_cost, assign_cost, time_limit=30)
        if milp_result is not None:
            load = np.zeros(n_fac)
            for c in range(n_cust):
                load[milp_result[c]] += demands[c]
            if np.all(load <= capacities + 1e-6):
                milp_cost = compute_cost(milp_result)
                if milp_cost < best_cost:
                    best_result = milp_result
                    best_cost = milp_cost

    # Multi-restart local search with different initializations
    rng = np.random.default_rng(42)
    num_restarts = 5

    for restart in range(num_restarts):
        if time.time() >= deadline - 10:
            break

        if restart == 0:
            # Use current best
            current = best_result.copy()
        else:
            # Random feasible initialization
            current = _random_feasible(demands, capacities, open_cost, assign_cost, rng)

        # Apply local search
        current = _local_search(demands, capacities, open_cost, assign_cost, current, max_iters=100)
        current_cost = compute_cost(current)

        if current_cost < best_cost:
            best_result = current
            best_cost = current_cost

        # Try swap local search on the best
        if time.time() < deadline - 5:
            current2 = _swap_local_search(demands, capacities, open_cost, assign_cost, best_result, max_iters=50)
            current2_cost = compute_cost(current2)
            if current2_cost < best_cost:
                best_result = current2
                best_cost = current2_cost

    # Final safety check
    result = best_result.astype(int)
    load = np.zeros(n_fac)
    for c in range(n_cust):
        load[result[c]] += demands[c]
    if np.any(load > capacities + 1e-6):
        # Fallback to greedy
        result = _greedy_assign(demands, capacities, open_cost, assign_cost)

    return result
# EVOLVE-BLOCK-END
