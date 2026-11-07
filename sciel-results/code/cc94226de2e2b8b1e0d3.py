"""Capacitated facility location: construction + local search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _construct(demands, capacities, open_cost, assign_cost):
    """Greedy construction: largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=float)
    order = np.argsort(-demands)
    for i in order:
        # Sort facilities by cost for this customer
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    return result, load


def _total_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost and check feasibility."""
    n_fac = len(capacities)
    load = np.zeros(n_fac, dtype=float)
    for i in range(len(result)):
        load[result[i]] += demands[i]
    if np.any(load > capacities + 1e-9):
        return np.inf
    open_mask = load > 0
    cost = np.sum(open_cost[open_mask])
    cost += np.sum(assign_cost[np.arange(len(result)), result])
    return cost


def _local_search(result, demands, capacities, open_cost, assign_cost, deadline):
    """Iterative local search: single moves + facility closure attempts."""
    n_cust = len(demands)
    n_fac = len(capacities)
    best_result = result.copy()
    best_cost = _total_cost(best_result, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return best_result, best_cost

    load = np.zeros(n_fac, dtype=float)
    for i in range(n_cust):
        load[best_result[i]] += demands[i]

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try single moves: move customer i to facility f if it reduces cost
        for i in range(n_cust):
            if time.time() >= deadline:
                break
            cur_f = best_result[i]
            d = demands[i]
            cur_cost = assign_cost[i, cur_f]
            # Try all other facilities
            for f in range(n_fac):
                if f == cur_f:
                    continue
                if load[f] + d > capacities[f] + 1e-9:
                    continue
                new_cost = assign_cost[i, f]
                # Cost delta: change in assignment cost + possible open cost change
                delta_assign = new_cost - cur_cost
                # If f was closed and we open it, add open_cost[f]
                delta_open = 0.0
                if load[f] == 0:
                    delta_open = open_cost[f]
                # If cur_f becomes empty after moving, we save open_cost[cur_f]
                delta_close = 0.0
                if load[cur_f] - d == 0:
                    delta_close = -open_cost[cur_f]
                total_delta = delta_assign + delta_open + delta_close
                if total_delta < -1e-9:
                    # Accept the move
                    load[cur_f] -= d
                    load[f] += d
                    best_result[i] = f
                    best_cost += total_delta
                    improved = True
                    break

        # Try facility closure: for each open facility, try to close it
        # by reassigning all its customers to other open facilities
        for f_close in range(n_fac):
            if time.time() >= deadline:
                break
            if load[f_close] == 0:
                continue
            # Get customers assigned to f_close
            custs = np.where(best_result == f_close)[0]
            if len(custs) == 0:
                continue
            # Try to reassign all to other open facilities
            new_load = load.copy()
            new_load[f_close] = 0.0
            new_result = best_result.copy()
            feasible = True
            total_new_assign = 0.0
            for i in custs:
                d = demands[i]
                # Find cheapest feasible facility (excluding f_close)
                best_f = -1
                best_c = np.inf
                for f in range(n_fac):
                    if f == f_close:
                        continue
                    if new_load[f] + d <= capacities[f] + 1e-9:
                        c = assign_cost[i, f]
                        if c < best_c:
                            best_c = c
                            best_f = f
                if best_f == -1:
                    feasible = False
                    break
                new_load[best_f] += d
                new_result[i] = best_f
                total_new_assign += best_c
            if not feasible:
                continue
            # Compute new total cost
            open_mask_new = new_load > 0
            new_total = np.sum(open_cost[open_mask_new]) + total_new_assign
            # Add assignment costs for customers NOT in custs (unchanged)
            other_custs = np.where(~np.isin(np.arange(n_cust), custs))[0]
            if len(other_custs) > 0:
                new_total += np.sum(assign_cost[other_custs, new_result[other_custs]])
            if new_total < best_cost - 1e-9:
                best_result = new_result
                best_cost = new_total
                load = new_load
                improved = True

    return best_result, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    """Assign each customer to a facility, minimizing total cost."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    # Construction
    result, load = _construct(demands, capacities, open_cost, assign_cost)

    # Check feasibility of construction
    if np.any(load > capacities + 1e-9):
        # Fallback: assign each customer to any feasible facility
        for i in range(n_cust):
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break

    # Local search
    result, best_cost = _local_search(result, demands, capacities, open_cost, assign_cost, deadline)

    # Final feasibility check
    load = np.zeros(n_fac, dtype=float)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    if np.any(load > capacities + 1e-9):
        # Should not happen, but return a safe fallback
        result, load = _construct(demands, capacities, open_cost, assign_cost)

    return result.astype(int)
# EVOLVE-BLOCK-END
