"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_total_cost(result, assign_cost, open_cost):
    """Compute total cost: open costs for used facilities + assignment costs."""
    n_cust = len(result)
    n_fac = len(open_cost)
    used = np.zeros(n_fac, dtype=bool)
    for i in range(n_cust):
        used[result[i]] = True
    total = np.sum(open_cost[used])
    total += np.sum(assign_cost[np.arange(n_cust), result])
    return total


def _greedy_fallback(demands, capacities, assign_cost):
    """Greedy fallback: largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    
    for i in np.argsort(-demands):
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            remaining = capacities - load
            f = np.argmax(remaining)
            load[f] += demands[i]
            result[i] = f
    
    return result.astype(int)


def _dual_guided_assign(demands, capacities, assign_cost, open_cost, lam, deadline):
    """
    Assignment guided by Lagrangian dual prices.
    For each customer, pick facility minimizing assign_cost[i,f] + lam[f]*demands[i].
    Then repair infeasibilities.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Compute effective cost with dual prices
    effective_cost = assign_cost + np.outer(demands, lam)
    
    # Initial assignment: each customer picks cheapest effective-cost facility
    result = np.argmin(effective_cost, axis=1)
    
    # Check feasibility and repair
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    # Repair: move customers from overloaded facilities
    for f in range(n_fac):
        if load[f] > capacities[f] + 1e-9:
            # Find customers at f, move the one with highest effective cost to another facility
            custs_at_f = [i for i in range(n_cust) if result[i] == f]
            # Sort by effective cost at f (highest first - most "expensive" to keep here)
            custs_at_f.sort(key=lambda i: effective_cost[i, f], reverse=True)
            
            for i in custs_at_f:
                if load[f] <= capacities[f] + 1e-9:
                    break
                # Find best alternative facility
                best_f = -1
                best_cost = np.inf
                for f2 in range(n_fac):
                    if f2 == f:
                        continue
                    if load[f2] + demands[i] <= capacities[f2] + 1e-9:
                        cost = effective_cost[i, f2]
                        if cost < best_cost:
                            best_cost = cost
                            best_f = f2
                if best_f >= 0:
                    load[f] -= demands[i]
                    load[best_f] += demands[i]
                    result[i] = best_f
                else:
                    # Force assign to facility with most remaining capacity
                    rem_cap = capacities - load
                    rem_cap[f] = -1  # Don't assign back to f
                    best_f = np.argmax(rem_cap)
                    load[f] -= demands[i]
                    load[best_f] += demands[i]
                    result[i] = best_f
    
    return result.astype(int)


def _subgradient_lagrangian(demands, capacities, assign_cost, open_cost, deadline):
    """
    Subgradient method for Lagrangian relaxation of capacity constraints.
    Returns dual prices lam that guide facility selection.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Start with zero dual prices
    lam = np.zeros(n_fac)
    best_lam = np.zeros(n_fac)
    
    # Adaptive step size
    step_scale = 1.0
    max_iter = 200
    iter_deadline = deadline - 5.0
    
    for it in range(max_iter):
        if time.time() > iter_deadline:
            break
        
        # Solve relaxed problem: for each customer, pick facility minimizing
        # assign_cost[i,f] + lam[f]*demands[i]
        effective_cost = assign_cost + np.outer(demands, lam)
        result = np.argmin(effective_cost, axis=1)
        
        # Compute load
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[result[i]] += demands[i]
        
        # Compute Lagrangian lower bound
        lb = 0.0
        used = np.zeros(n_fac, dtype=bool)
        for i in range(n_cust):
            lb += assign_cost[i, result[i]]
            used[result[i]] = True
        lb += np.sum(open_cost[used])
        for f in range(n_fac):
            lb += lam[f] * (load[f] - capacities[f])
        
        # Subgradient: g[f] = load[f] - capacities[f]
        g = load - capacities
        
        # Update step size (adaptive)
        gap_estimate = np.max(np.abs(g))
        if gap_estimate < 1e-6:
            break
        
        # Polyak-style step
        step = step_scale / (np.linalg.norm(g) + 1e-9)
        lam_new = np.maximum(0, lam + step * g)
        
        # Check if improved
        if np.linalg.norm(lam_new - lam) < 1e-8:
            lam = lam_new
            break
        
        lam = lam_new
        step_scale *= 0.995  # Gradually reduce step size
        
        best_lam = lam.copy()
    
    return best_lam


def _local_search_1move(result, demands, capacities, assign_cost, open_cost, deadline):
    """Local search: try moving each customer to a better facility."""
    n_cust = len(result)
    n_fac = len(capacities)
    
    result = result.copy()
    load = np.zeros(n_fac)
    cust_count = np.zeros(n_fac, dtype=int)
    for i in range(n_cust):
        load[result[i]] += demands[i]
        cust_count[result[i]] += 1
    
    best_cost = _compute_total_cost(result, assign_cost, open_cost)
    improved = True
    
    sorted_facs = np.argsort(assign_cost, axis=1)
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_cust):
            if time.time() > deadline:
                break
            current_f = result[i]
            for f in sorted_facs[i]:
                if f == current_f:
                    continue
                if load[f] + demands[i] > capacities[f] + 1e-9:
                    continue
                
                delta_assign = assign_cost[i, f] - assign_cost[i, current_f]
                
                delta_open = 0.0
                if cust_count[current_f] == 1:
                    delta_open -= open_cost[current_f]
                if cust_count[f] == 0:
                    delta_open += open_cost[f]
                
                total_delta = delta_assign + delta_open
                if total_delta < -1e-9:
                    load[current_f] -= demands[i]
                    load[f] += demands[i]
                    cust_count[current_f] -= 1
                    cust_count[f] += 1
                    result[i] = f
                    best_cost += total_delta
                    improved = True
                    break
    
    return result, best_cost


def _local_search_2swap(result, demands, capacities, assign_cost, open_cost, deadline):
    """Local search: try swapping pairs of customers between facilities."""
    n_cust = len(result)
    n_fac = len(capacities)
    
    result = result.copy()
    load = np.zeros(n_fac)
    cust_count = np.zeros(n_fac, dtype=int)
    for i in range(n_cust):
        load[result[i]] += demands[i]
        cust_count[result[i]] += 1
    
    best_cost = _compute_total_cost(result, assign_cost, open_cost)
    improved = True
    
    fac_to_custs = {f: [] for f in range(n_fac)}
    for i in range(n_cust):
        fac_to_custs[result[i]].append(i)
    
    while improved and time.time() < deadline:
        improved = False
        for f1 in range(n_fac):
            if time.time() > deadline:
                break
            for f2 in range(f1 + 1, n_fac):
                if time.time() > deadline:
                    break
                custs1 = list(fac_to_custs[f1])
                custs2 = list(fac_to_custs[f2])
                if not custs1 or not custs2:
                    continue
                for i in custs1:
                    if time.time() > deadline:
                        break
                    for j in custs2:
                        if load[f1] - demands[i] + demands[j] > capacities[f1] + 1e-6:
                            continue
                        if load[f2] - demands[j] + demands[i] > capacities[f2] + 1e-6:
                            continue
                        
                        delta = (assign_cost[i, f2] + assign_cost[j, f1]) - \
                                (assign_cost[i, f1] + assign_cost[j, f2])
                        
                        count_f1 = len(custs1)
                        count_f2 = len(custs2)
                        delta_open = 0.0
                        if count_f1 == 1:
                            delta_open -= open_cost[f1]
                        if count_f2 == 1:
                            delta_open -= open_cost[f2]
                        
                        total_delta = delta + delta_open
                        if total_delta < -1e-9:
                            load[f1] += demands[j] - demands[i]
                            load[f2] += demands[i] - demands[j]
                            result[i] = f2
                            result[j] = f1
                            best_cost += total_delta
                            improved = True
                            fac_to_custs[f1].remove(i)
                            fac_to_custs[f2].remove(j)
                            fac_to_custs[f2].append(i)
                            fac_to_custs[f1].append(j)
                            break
                    else:
                        continue
                    break
                if improved and time.time() > deadline:
                    break
    
    return result, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each of 100 customers to one of 20 facilities for capacitated facility location.
    Uses Lagrangian relaxation with subgradient optimization for warm start,
    followed by local search with 1-moves and 2-swaps.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Phase 1: Lagrangian relaxation to find dual prices
    lam = _subgradient_lagrangian(demands, capacities, assign_cost, open_cost, deadline)
    
    # Phase 2: Dual-guided assignment
    result = _dual_guided_assign(demands, capacities, assign_cost, open_cost, lam, deadline)
    
    # Phase 3: Local search with 1-moves
    result, cost = _local_search_1move(result, demands, capacities, assign_cost, open_cost, deadline)
    
    # Phase 4: Local search with 2-swaps
    result, cost = _local_search_2swap(result, demands, capacities, assign_cost, open_cost, deadline)
    
    # Phase 5: Final 1-move polish
    result, cost = _local_search_1move(result, demands, capacities, assign_cost, open_cost, deadline)
    
    # Verify feasibility
    load = np.zeros(n_fac)
    for i in range(n_cust):
        if result[i] < 0 or result[i] >= n_fac:
            result[i] = 0
        load[result[i]] += demands[i]
    
    if np.any(load > capacities + 1e-6):
        result = _greedy_fallback(demands, capacities, assign_cost)
    
    return result.astype(int)
# EVOLVE-BLOCK-END
