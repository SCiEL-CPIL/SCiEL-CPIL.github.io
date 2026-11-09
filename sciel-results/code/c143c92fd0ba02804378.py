# EVOLVE-BLOCK-START
import numpy as np
import time


def _true_obj(assign_arr, demands, capacities, open_cost, assign_cost):
    """Compute the objective: open cost + assignment cost."""
    n_fac = len(capacities)
    load = np.bincount(assign_arr, minlength=n_fac)
    # Check capacity
    if np.any(load > capacities):
        return np.inf
    opened = load > 0
    open_part = open_cost[opened].sum()
    assign_part = assign_cost[np.arange(len(assign_arr)), assign_arr].sum()
    return open_part + assign_part


def _greedy_seed(demands, capacities, open_cost, assign_cost, order, open_subset):
    """Greedy assignment: assign customers in given order to cheapest feasible
    facility among open_subset. Returns assignment array or None if infeasible."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac, dtype=float)
    for i in order:
        d = demands[i]
        best_f = -1
        best_c = np.inf
        for f in open_subset:
            if load[f] + d <= capacities[f]:
                c = assign_cost[i, f]
                if c < best_c:
                    best_c = c
                    best_f = f
        if best_f == -1:
            return None
        load[best_f] += d
        result[i] = best_f
    return result


def _local_search(assign_arr, demands, capacities, open_cost, assign_cost, deadline):
    """Local search: try moving each customer to a different facility,
    accounting for opening/closing cost deltas."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = assign_arr.copy()
    load = np.bincount(result, minlength=n_fac).astype(float)
    cur_obj = _true_obj(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(cur_obj):
        return result

    improved = True
    while improved:
        improved = False
        # Try moving customers in order of largest demand first for faster convergence
        for i in np.argsort(-demands):
            if time.time() > deadline:
                break
            d = demands[i]
            cur_f = result[i]
            cur_cost_i = assign_cost[i, cur_f]
            # Try all other facilities
            for f in range(n_fac):
                if f == cur_f:
                    continue
                # Check capacity
                if load[f] + d > capacities[f]:
                    continue
                new_cost_i = assign_cost[i, f]
                # Compute delta in assignment cost
                delta_assign = new_cost_i - cur_cost_i
                # Compute delta in open cost
                # cur_f loses one customer, f gains one
                cur_load_cur = load[cur_f]
                new_load_cur = cur_load_cur - d
                cur_load_f = load[f]
                new_load_f = cur_load_f + d

                delta_open = 0.0
                # If cur_f becomes empty, we save its open cost
                if new_load_cur == 0 and cur_load_cur > 0:
                    delta_open -= open_cost[cur_f]
                # If f was empty and becomes non-empty, we pay its open cost
                if cur_load_f == 0 and new_load_f > 0:
                    delta_open += open_cost[f]

                total_delta = delta_assign + delta_open
                if total_delta < -1e-9:
                    # Apply the move
                    load[cur_f] = new_load_cur
                    load[f] = new_load_f
                    result[i] = f
                    cur_obj += total_delta
                    improved = True
                    break
            if time.time() > deadline:
                break
    return result


def _multi_start_local_search(demands, capacities, open_cost, assign_cost, deadline):
    """Run multiple greedy seeds with different orderings and facility subsets,
    then local search each, keep the best."""
    n_cust = len(demands)
    n_fac = len(capacities)

    # Precompute some orderings
    orderings = [
        np.argsort(-demands),  # largest demand first
        np.argsort(demands),   # smallest demand first
        np.arange(n_cust),     # natural order
    ]
    # Add a couple of randomized orderings using a fixed seed for reproducibility
    rng = np.random.RandomState(42)
    orderings.append(rng.permutation(n_cust))
    orderings.append(rng.permutation(n_cust))

    # Facility subsets: all facilities, and some random subsets
    all_facs = np.arange(n_fac)
    subsets = [all_facs]
    # Add a few random subsets of size ~ n_fac * 0.7 to 0.9
    for size_frac in [0.7, 0.8, 0.9]:
        k = max(1, int(n_fac * size_frac))
        subsets.append(rng.choice(n_fac, size=k, replace=False))

    best_assign = None
    best_obj = np.inf

    for order in orderings:
        for subset in subsets:
            if time.time() > deadline:
                break
            seed = _greedy_seed(demands, capacities, open_cost, assign_cost, order, subset)
            if seed is None:
                continue
            # Local search with remaining time budget
            ls_deadline = deadline - 0.5  # safety margin
            refined = _local_search(seed, demands, capacities, open_cost, assign_cost, ls_deadline)
            obj = _true_obj(refined, demands, capacities, open_cost, assign_cost)
            if obj < best_obj:
                best_obj = obj
                best_assign = refined.copy()
        if time.time() > deadline:
            break

    if best_assign is None:
        # Fallback: simple greedy with all facilities
        best_assign = _greedy_seed(demands, capacities, open_cost, assign_cost,
                                   np.argsort(-demands), all_facs)
        if best_assign is None:
            # Last resort: assign everyone to facility 0 (may be invalid but avoid crash)
            best_assign = np.zeros(n_cust, dtype=int)
    return best_assign


def assign(demands, capacities, open_cost, assign_cost):
    start = time.time()
    deadline = start + 145.0  # 5s safety margin before 150s kill

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    result = _multi_start_local_search(demands, capacities, open_cost, assign_cost, deadline)

    # Final validity check
    load = np.bincount(result, minlength=len(capacities))
    if np.any(load > capacities):
        # Fallback to simple greedy
        result = np.zeros(len(demands), dtype=int)
        load = np.zeros(len(capacities))
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break

    return result.astype(int)
# EVOLVE-BLOCK-END
