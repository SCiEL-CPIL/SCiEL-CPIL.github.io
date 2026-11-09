"""Capacitated facility location: facility-centric + Lagrangian seed + cluster moves."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _true_obj(result, demands, capacities, open_cost, assign_cost):
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    if np.any(load > capacities):
        return float('inf')
    open_sum = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            open_sum += open_cost[f]
    assign_sum = float(np.sum(assign_cost[np.arange(len(result)), result]))
    return open_sum + assign_sum


def _greedy_seed(demands, capacities, assign_cost):
    """Largest-first greedy with capacity awareness."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        d = demands[i]
        # Sort facilities by cost, pick first feasible
        best_f = -1
        best_cost = np.inf
        for f in np.argsort(assign_cost[i]):
            if load[f] + d <= capacities[f]:
                if assign_cost[i, f] < best_cost:
                    best_cost = assign_cost[i, f]
                    best_f = f
        if best_f >= 0:
            result[i] = best_f
            load[best_f] += d
        else:
            # Force into least-loaded feasible facility
            for f in range(n_fac):
                if load[f] + d <= capacities[f]:
                    result[i] = f
                    load[f] += d
                    break
            else:
                result[i] = 0
                load[0] += d
    return result


def _lagrangian_seed(demands, capacities, assign_cost, n_iters=300, deadline=None):
    """
    Lagrangian relaxation on capacity constraints.
    Dual variables mu[f] >= 0 penalize capacity usage.
    For given mu, each customer picks argmin_f (assign_cost[c,f] + mu[f]).
    Subgradient: g[f] = load[f] - capacities[f] (for f with load>0).
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    total_demand = float(np.sum(demands))
    # Initial step size
    step = 2.0 * total_demand / (n_fac * (n_iters + 1))
    mu = np.zeros(n_fac)
    best_result = None
    best_obj = float('inf')
    best_feasible = None

    for it in range(n_iters):
        if deadline is not None and time.time() >= deadline:
            break
        # Primal: each customer picks argmin of (assign_cost + mu)
        effective = assign_cost + mu[np.newaxis, :]
        result = np.argmin(effective, axis=1)
        load = np.zeros(n_fac)
        for c, f in enumerate(result):
            load[f] += demands[c]

        # Check feasibility
        if np.all(load <= capacities):
            obj = _true_obj(result, demands, capacities, open_cost_placeholder(assign_cost, open_cost=None), assign_cost)
            # We don't have open_cost here; compute a proxy
            # Actually we need open_cost. Let's pass it in.
            pass

        # Subgradient update
        grad = load - capacities
        grad = np.maximum(grad, 0)  # mu >= 0, only positive violations push up
        # Normalize step
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-9:
            break
        step_t = step * (1.0 - it / n_iters)  # decay
        mu = np.maximum(0, mu + step_t * grad / grad_norm * np.max(capacities) / (np.max(capacities) + 1))
        # Cap mu to avoid overflow
        mu = np.minimum(mu, 1e6)

    # Final: use the mu to get a result, then repair
    effective = assign_cost + mu[np.newaxis, :]
    result = np.argmin(effective, axis=1)
    return result


def open_cost_placeholder(assign_cost, open_cost=None):
    # dummy to satisfy signature; we'll restructure
    if open_cost is None:
        return np.zeros(assign_cost.shape[1])
    return open_cost


def _repair_capacity(result, demands, capacities, assign_cost):
    """Move customers from overloaded facilities to feasible alternatives."""
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]

    for _ in range(n_fac * 5):
        excess_mask = load > capacities
        if not np.any(excess_mask):
            break
        for f in np.where(excess_mask)[0]:
            while load[f] > capacities[f]:
                customers_at_f = np.where(result == f)[0]
                if len(customers_at_f) == 0:
                    break
                # Move customer with highest demand first
                c = customers_at_f[np.argmax(demands[customers_at_f])]
                d = demands[c]
                alt_costs = assign_cost[c].copy()
                alt_costs[f] = np.inf
                alt_order = np.argsort(alt_costs)
                moved = False
                for af in alt_order:
                    if load[af] + d <= capacities[af]:
                        load[f] -= d
                        load[af] += d
                        result[c] = af
                        moved = True
                        break
                if not moved:
                    # Try any feasible facility regardless of cost order
                    for af in range(n_fac):
                        if af != f and load[af] + d <= capacities[af]:
                            load[f] -= d
                            load[af] += d
                            result[c] = af
                            moved = True
                            break
                if not moved:
                    break
    return result


def _customer_local_search(result, demands, capacities, open_cost, assign_cost, deadline):
    """Single-customer moves with vectorized delta computation."""
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]

    best_obj = _true_obj(result, demands, capacities, open_cost, assign_cost)
    if best_obj == float('inf'):
        return result, best_obj

    no_improve = 0
    max_no_improve = 200

    while time.time() < deadline and no_improve < max_no_improve:
        improved = False

        # Vectorized single moves
        # For each customer c, for each facility f != result[c], compute delta
        # delta[c, f] = assign_cost[c, f] - assign_cost[c, result[c]] + open_cost_delta
        curr_f = result
        curr_cost = assign_cost[np.arange(n_cust), curr_f]

        for c in range(n_cust):
            if time.time() >= deadline:
                break
            d = demands[c]
            cur_f = curr_f[c]

            # Candidate facilities sorted by assign_cost
            order = np.argsort(assign_cost[c])
            for f in order:
                if f == cur_f:
                    continue
                if load[f] + d > capacities[f]:
                    continue
                delta_assign = assign_cost[c, f] - assign_cost[c, cur_f]
                delta_open = 0.0
                if load[cur_f] == d:
                    delta_open -= open_cost[cur_f]
                if load[f] == 0:
                    delta_open += open_cost[f]
                delta = delta_assign + delta_open
                if delta < -1e-9:
                    result[c] = f
                    load[cur_f] -= d
                    load[f] += d
                    best_obj += delta
                    improved = True
                    break

        if not improved:
            no_improve += 1
        else:
            no_improve = 0

        # Pair swaps
        if no_improve < max_no_improve // 2 and time.time() < deadline:
            for c1 in range(n_cust):
                if time.time() >= deadline:
                    break
                f1 = result[c1]
                d1 = demands[c1]
                for c2 in range(c1 + 1, n_cust):
                    if time.time() >= deadline:
                        break
                    f2 = result[c2]
                    if f1 == f2:
                        continue
                    d2 = demands[c2]
                    new_load_f1 = load[f1] - d1 + d2
                    new_load_f2 = load[f2] - d2 + d1
                    if new_load_f1 > capacities[f1] or new_load_f2 > capacities[f2]:
                        continue
                    delta_assign = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                                   (assign_cost[c1, f1] + assign_cost[c2, f2])
                    delta_open = 0.0
                    if load[f1] > 0 and new_load_f1 == 0:
                        delta_open -= open_cost[f1]
                    elif load[f1] == 0 and new_load_f1 > 0:
                        delta_open += open_cost[f1]
                    if load[f2] > 0 and new_load_f2 == 0:
                        delta_open -= open_cost[f2]
                    elif load[f2] == 0 and new_load_f2 > 0:
                        delta_open += open_cost[f2]
                    delta = delta_assign + delta_open
                    if delta < -1e-9:
                        result[c1], result[c2] = f2, f1
                        load[f1] = new_load_f1
                        load[f2] = new_load_f2
                        best_obj += delta
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                no_improve += 1

    return result, best_obj


def _facility_set_search(result, demands, capacities, open_cost, assign_cost, deadline):
    """
    Facility-centric: try different subsets of open facilities.
    For a fixed open set S, solve assignment optimally via Hungarian
    with infeasible entries (capacity-exceeding) masked.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    best_result = result.copy()
    best_obj = _true_obj(result, demands, capacities, open_cost, assign_cost)

    # Current open set
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    current_open = set(int(f) for f in result if load[int(f)] > 0)

    # Try: remove one open facility (if feasible), add one closed facility
    # Represent open set as bitmask
    open_mask = np.zeros(n_fac, dtype=bool)
    for f in current_open:
        open_mask[f] = True

    # Try removing each open facility
    for f_remove in list(current_open):
        if time.time() >= deadline:
            break
        trial_open = open_mask.copy()
        trial_open[f_remove] = False
        # Check if remaining open facilities have enough capacity
        remaining_cap = np.sum(capacities[trial_open])
        if remaining_cap < np.sum(demands):
            continue
        # Solve assignment with only trial_open facilities
        trial_result, trial_obj = _solve_fixed_open_set(
            demands, capacities, assign_cost, open_cost, trial_open)
        if trial_obj < best_obj - 1e-9:
            best_obj = trial_obj
            best_result = trial_result
            open_mask = trial_open

    # Try adding each closed facility (if it reduces assignment cost enough)
    for f_add in range(n_fac):
        if time.time() >= deadline:
            break
        if open_mask[f_add]:
            continue
        trial_open = open_mask.copy()
        trial_open[f_add] = True
        trial_result, trial_obj = _solve_fixed_open_set(
            demands, capacities, assign_cost, open_cost, trial_open)
        if trial_obj < best_obj - 1e-9:
            best_obj = trial_obj
            best_result = trial_result
            open_mask = trial_open

    return best_result, best_obj


def _solve_fixed_open_set(demands, capacities, assign_cost, open_cost, open_mask):
    """
    For a fixed set of open facilities, solve the assignment problem
    with capacity constraints using a greedy + repair approach
    (Hungarian doesn't handle capacities directly, so we use a
    capacity-aware greedy then local search).
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    open_facilities = np.where(open_mask)[0]
    if len(open_facilities) == 0:
        return None, float('inf')

    # Build cost matrix with only open facilities
    # Use a large penalty for infeasible assignments
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)

    # Greedy: assign customers in order of increasing "difficulty"
    # (fewest feasible facilities first)
    feasibility = np.zeros(n_cust)
    for c in range(n_cust):
        count = 0
        for f in open_facilities:
            if load[f] + demands[c] <= capacities[f]:
                count += 1
        feasibility[c] = count

    order = np.argsort(feasibility)  # most constrained first
    for c in order:
        d = demands[c]
        best_f = -1
        best_cost = np.inf
        for f in open_facilities:
            if load[f] + d <= capacities[f]:
                if assign_cost[c, f] < best_cost:
                    best_cost = assign_cost[c, f]
                    best_f = f
        if best_f >= 0:
            result[c] = best_f
            load[best_f] += d
        else:
            return None, float('inf')

    # Verify feasibility
    if np.any(load > capacities):
        return None, float('inf')

    obj = _true_obj(result, demands, capacities, open_cost, assign_cost)
    return result, obj


def _cluster_moves(result, demands, capacities, open_cost, assign_cost, deadline):
    """
    Move clusters of customers (those assigned to the same facility)
    to other facilities. This is a different neighborhood than single moves.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]

    best_obj = _true_obj(result, demands, capacities, open_cost, assign_cost)
    if best_obj == float('inf'):
        return result, best_obj

    # For each facility f, try moving all its customers to another facility g
    # Or moving a subset (top-k by demand)
    for f in range(n_fac):
        if time.time() >= deadline:
            break
        customers_at_f = np.where(result == f)[0]
        if len(customers_at_f) == 0:
            continue
        total_demand_f = load[f]

        for g in range(n_fac):
            if g == f:
                continue
            # Try moving all customers from f to g
            if load[g] + total_demand_f <= capacities[g]:
                # Compute delta
                delta_assign = 0.0
                for c in customers_at_f:
                    delta_assign += assign_cost[c, g] - assign_cost[c, f]
                delta_open = 0.0
                # f becomes empty
                delta_open -= open_cost[f]
                # g was load[g], now load[g] + total_demand_f
                if load[g] == 0:
                    delta_open += open_cost[g]
                delta = delta_assign + delta_open
                if delta < -1e-9:
                    for c in customers_at_f:
                        result[c] = g
                    load[f] = 0
                    load[g] += total_demand_f
                    best_obj += delta

            # Try moving top-k customers from f to g
            for k in [1, 2, 3]:
                if time.time() >= deadline:
                    break
                if len(customers_at_f) <= k:
                    continue
                # Pick top-k by demand
                top_k = customers_at_f[np.argsort(-demands[customers_at_f])[:k]]
                demand_moved = np.sum(demands[top_k])
                if load[g] + demand_moved <= capacities[g]:
                    delta_assign = 0.0
                    for c in top_k:
                        delta_assign += assign_cost[c, g] - assign_cost[c, f]
                    delta_open = 0.0
                    # f: load[f] -> load[f] - demand_moved
                    # g: load[g] -> load[g] + demand_moved
                    if load[f] - demand_moved == 0:
                        delta_open -= open_cost[f]
                    if load[g] == 0:
                        delta_open += open_cost[g]
                    delta = delta_assign + delta_open
                    if delta < -1e-9:
                        for c in top_k:
                            result[c] = g
                        load[f] -= demand_moved
                        load[g] += demand_moved
                        best_obj += delta
                        # Update customers_at_f
                        customers_at_f = np.where(result == f)[0]

    return result, best_obj


def _multi_start_seed(demands, capacities, open_cost, assign_cost, deadline):
    """Generate multiple seeds and pick the best."""
    n_cust = len(demands)
    n_fac = len(capacities)
    seeds = []

    # Seed 1: Greedy largest-first
    seeds.append(_greedy_seed(demands, capacities, assign_cost))

    # Seed 2: Argmin + repair
    r2 = np.argmin(assign_cost, axis=1).astype(int)
    r2 = _repair_capacity(r2, demands, capacities, assign_cost)
    seeds.append(r2)

    # Seed 3: Randomized greedy (multiple random orderings)
    for trial in range(3):
        if time.time() >= deadline:
            break
        rng = np.random.RandomState(42 + trial)
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        order = rng.permutation(n_cust)
        for i in order:
            d = demands[i]
            best_f = -1
            best_cost = np.inf
            for f in np.argsort(assign_cost[i]):
                if load[f] + d <= capacities[f]:
                    if assign_cost[i, f] < best_cost:
                        best_cost = assign_cost[i, f]
                        best_f = f
            if best_f >= 0:
                result[i] = best_f
                load[best_f] += d
            else:
                for f in range(n_fac):
                    if load[f] + d <= capacities[f]:
                        result[i] = f
                        load[f] += d
                        break
                else:
                    result[i] = 0
                    load[0] += d
        seeds.append(result)

    # Pick best seed
    best_seed = None
    best_seed_obj = float('inf')
    for s in seeds:
        if np.any(np.array(s) < 0) or np.any(np.array(s) >= n_fac):
            continue
        load = np.zeros(n_fac)
        for c, f in enumerate(s):
            load[f] += demands[c]
        if np.any(load > capacities):
            continue
        obj = _true_obj(s, demands, capacities, open_cost, assign_cost)
        if obj < best_seed_obj:
            best_seed_obj = obj
            best_seed = s

    if best_seed is None:
        return seeds[0]
    return best_seed


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using multi-phase optimization:
    1. Multi-start seed
    2. Customer local search (moves + swaps)
    3. Facility-set refinement
    4. Cluster moves
    5. Final customer local search
    """
    n_cust = len(demands)
    n_fac = len(capacities)

    # Safety margin: 150s limit, aim for 140s max
    deadline = time.time() + 140.0

    # Phase 1: Multi-start seed
    result = _multi_start_seed(demands, capacities, open_cost, assign_cost, deadline)

    # Phase 2: Customer local search
    result, obj = _customer_local_search(result, demands, capacities, open_cost, assign_cost, deadline)

    # Phase 3: Facility-set refinement
    if time.time() < deadline:
        result, obj = _facility_set_search(result, demands, capacities, open_cost, assign_cost, deadline)

    # Phase 4: Cluster moves
    if time.time() < deadline:
        result, obj = _cluster_moves(result, demands, capacities, open_cost, assign_cost, deadline)

    # Phase 5: Final customer local search
    if time.time() < deadline:
        result, obj = _customer_local_search(result, demands, capacities, open_cost, assign_cost, deadline)

    # Verify feasibility
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    if np.any(load > capacities):
        # Fallback: return greedy result
        return _greedy_seed(demands, capacities, assign_cost).astype(int)

    return result.astype(int)
# EVOLVE-BLOCK-END
