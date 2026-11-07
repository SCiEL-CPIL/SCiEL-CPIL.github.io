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


def _greedy_assign(demands, capacities, assign_cost):
    """Greedy: largest demand first, cheapest feasible facility."""
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


def _lagrangian_dual(demands, capacities, assign_cost, open_cost, max_iter=200):
    """
    Lagrangian relaxation: find dual prices for capacity constraints.
    Returns dual prices lambda[f] >= 0 that guide facility selection.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Start with zero dual prices
    lam = np.zeros(n_fac)
    
    for it in range(max_iter):
        # Solve relaxed problem: for each customer, pick facility minimizing
        # assign_cost[i,f] + lam[f] * demands[i] (ignoring capacity)
        # Plus open cost if facility becomes "used" in a heuristic sense
        # Simplified: just minimize assign_cost[i,f] + lam[f] * demands[i]
        adjusted_cost = assign_cost + np.outer(demands, lam)
        best_facs = np.argmin(adjusted_cost, axis=1)
        
        # Compute resulting loads
        loads = np.zeros(n_fac)
        for i in range(n_cust):
            loads[best_facs[i]] += demands[i]
        
        # Subgradient: loads - capacities
        subgrad = loads - capacities
        
        # Check convergence
        if np.all(subgrad <= 1e-6):
            break
        
        # Update dual prices (projected subgradient)
        # Step size: use diminishing step
        step = 2.0 / (it + 1)
        lam_new = lam + step * subgrad
        lam = np.maximum(lam_new, 0)
    
    return lam


def _dual_guided_assign(demands, capacities, assign_cost, lam):
    """
    Assign customers using dual prices to guide facility selection.
    Larger demand customers first, using adjusted cost = assign_cost + lam*demand.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    
    adjusted_cost = assign_cost + np.outer(demands, lam)
    
    for i in np.argsort(-demands):
        fac_order = np.argsort(adjusted_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # Fallback: find feasible facility with most remaining capacity
            remaining = capacities - load
            f = np.argmax(remaining)
            load[f] += demands[i]
            result[i] = f
    
    return result.astype(int)


def _local_search(result, demands, capacities, assign_cost, open_cost, deadline):
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


def _pairwise_swap(result, demands, capacities, assign_cost, open_cost, deadline):
    """Try pairwise swaps between customers at different facilities."""
    n_cust = len(result)
    n_fac = len(capacities)
    
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
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


def _final_polish(result, demands, capacities, assign_cost, open_cost, deadline):
    """Final polishing: multiple passes of local search with early termination."""
    n_cust = len(result)
    n_fac = len(capacities)
    
    result = result.copy()
    load = np.zeros(n_fac)
    cust_count = np.zeros(n_fac, dtype=int)
    for i in range(n_cust):
        load[result[i]] += demands[i]
        cust_count[result[i]] += 1
    
    best_cost = _compute_total_cost(result, assign_cost, open_cost)
    sorted_facs = np.argsort(assign_cost, axis=1)
    
    max_passes = 10
    for pass_num in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n_cust):
            if time.time() > deadline:
                break
            current_f = result[i]
            for f in sorted_facs[i][:15]:  # Check top 15 candidates
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
        if not improved:
            break
    
    return result, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each of 100 customers to one of 20 facilities for capacitated facility location.
    Uses Lagrangian dual-guided initialization, local search, pairwise swaps, and final polish.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Phase 1: Lagrangian dual to get capacity prices
    lam = _lagrangian_dual(demands, capacities, assign_cost, open_cost, max_iter=200)
    
    # Phase 2: Dual-guided greedy initialization
    result = _dual_guided_assign(demands, capacities, assign_cost, lam)
    
    # Phase 3: Local search
    result, cost = _local_search(result, demands, capacities, assign_cost, open_cost, deadline)
    
    # Phase 4: Pairwise swap local search
    result, cost = _pairwise_swap(result, demands, capacities, assign_cost, open_cost, deadline)
    
    # Phase 5: Final polish
    result, cost = _final_polish(result, demands, capacities, assign_cost, open_cost, deadline)
    
    # Verify feasibility
    load = np.zeros(n_fac)
    for i in range(n_cust):
        if result[i] < 0 or result[i] >= n_fac:
            result[i] = 0
        load[result[i]] += demands[i]
    
    if np.any(load > capacities + 1e-6):
        # Fallback to standard greedy if infeasible
        result = _greedy_assign(demands, capacities, assign_cost)
    
    return result.astype(int)
# EVOLVE-BLOCK-END
