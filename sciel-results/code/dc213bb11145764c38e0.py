"""Capacitated Facility Location – Lagrangian relaxation + vectorized local search.

Strategy:
  1. Lagrangian relaxation: dualize capacity constraints, solve decomposed
     subproblems with open-cost-aware heuristic, subgradient ascent.
  2. Feasibility repair from relaxed solutions.
  3. Fast vectorized local search (single-move) with pair-exchange polish.
  4. Multiple diverse restarts within time budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _lagrangian_subproblem(demands, capacities, open_cost, assign_cost, lam):
    """Solve Lagrangian subproblem with open-cost-aware greedy.
    
    For each facility, compute total adjusted cost including open cost
    amortized over expected customers. Use a two-pass approach:
    1. Assign customers to cheapest adjusted facility
    2. Re-evaluate with open costs properly accounted
    """
    n_cust, n_fac = assign_cost.shape
    adj_cost = assign_cost + lam[None, :] * demands[:, None]
    
    # Pass 1: initial assignment ignoring open costs
    result = np.argmin(adj_cost, axis=1)
    
    # Pass 2: refine by considering open costs
    # Compute load and counts
    load = np.zeros(n_fac)
    counts = np.zeros(n_fac)
    np.add.at(load, result, demands)
    np.add.at(counts, result, 1)
    
    # For facilities with no customers, add open cost to their column
    # Then re-assign customers that might benefit from switching
    adj_with_open = adj_cost.copy()
    for j in range(n_fac):
        if counts[j] == 0:
            adj_with_open[:, j] += open_cost[j]
    
    # Re-assign using adjusted costs with open costs
    new_result = np.argmin(adj_with_open, axis=1)
    
    # Check if this improves the objective
    def _eval(res):
        ld = np.zeros(n_fac)
        ct = np.zeros(n_fac)
        np.add.at(ld, res, demands)
        np.add.at(ct, res, 1)
        cost = 0.0
        for i in range(n_cust):
            cost += adj_cost[i, res[i]]
        for j in range(n_fac):
            if ct[j] > 0:
                cost += open_cost[j]
        return cost, ld, ct
    
    cost1, _, _ = _eval(result)
    cost2, load2, counts2 = _eval(new_result)
    
    if cost2 < cost1:
        result = new_result
        load = load2
        counts = counts2
    
    # Dual objective
    dual_obj = 0.0
    for i in range(n_cust):
        dual_obj += adj_cost[i, result[i]]
    for j in range(n_fac):
        if counts[j] > 0:
            dual_obj += open_cost[j]
    dual_obj -= np.dot(lam, capacities)
    
    return result, dual_obj, load, counts

def _subgradient_lagrangian(demands, capacities, open_cost, assign_cost, 
                             max_iter=200, time_limit=30.0):
    """Subgradient optimization of Lagrangian dual.
    
    Returns best feasible solution found and its cost.
    """
    n_cust, n_fac = assign_cost.shape
    start_time = time.time()
    
    # Initialize multipliers
    lam = np.zeros(n_fac)
    best_result = None
    best_feasible_cost = np.inf
    
    # Step size parameters
    best_gap = np.inf
    step_scale = 1.0
    best_result_relaxed = None
    best_dual_obj = -np.inf
    patience = 0
    max_patience = 50
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        result, dual_obj, load, counts = _lagrangian_subproblem(
            demands, capacities, open_cost, assign_cost, lam)
        
        # Track best dual bound
        if dual_obj > best_dual_obj:
            best_dual_obj = dual_obj
            best_result_relaxed = result.copy()
            patience = 0
        else:
            patience += 1
        
        # Check feasibility of relaxed solution
        if np.all(load <= capacities + 1e-9):
            true_cost = _compute_true_cost(demands, capacities, open_cost, assign_cost, result)
            if true_cost < best_feasible_cost:
                best_feasible_cost = true_cost
                best_result = result.copy()
        
        # Subgradient: load - capacities (positive where overloaded)
        grad = load - capacities
        
        # Gap for adaptive step size
        if best_result is not None:
            gap = best_feasible_cost - best_dual_obj
        else:
            gap = abs(best_dual_obj) if best_dual_obj != 0 else 1.0
        
        if gap > 1e-6:
            best_gap = min(best_gap, gap)
        
        # Polyak step size with decay
        step = step_scale * best_gap / (np.dot(grad, grad) + 1e-12)
        step = min(step, 0.5)  # Cap step size
        
        # Update multipliers
        lam_new = lam + step * grad
        lam = np.maximum(lam_new, 0.0)
        
        # Decay step scale
        if it % 30 == 0 and it > 0:
            step_scale *= 0.8
        
        # Early termination if no progress
        if patience > max_patience:
            break
    
    # Try the best relaxed solution as a starting point
    if best_result_relaxed is not None:
        result = best_result_relaxed.copy()
        result = _repair_feasibility(demands, capacities, result)
        cost = _compute_true_cost(demands, capacities, open_cost, assign_cost, result)
        if cost < best_feasible_cost:
            best_feasible_cost = cost
            best_result = result.copy()
    
    if best_result is None:
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost)
        best_feasible_cost = _compute_true_cost(demands, capacities, open_cost, assign_cost, best_result)
    
    return best_result, best_feasible_cost, best_dual_obj

def _repair_feasibility(demands, capacities, result):
    """Repair an infeasible assignment by moving customers from overloaded facilities."""
    n_cust, n_fac = len(demands), len(capacities)
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    # Iteratively fix overloaded facilities
    for _ in range(20):
        overloaded = np.where(load > capacities + 1e-9)[0]
        if len(overloaded) == 0:
            break
        for j in overloaded:
            if load[j] <= capacities[j] + 1e-9:
                continue
            # Get customers at facility j, sort by demand ascending
            custs = np.where(result == j)[0]
            custs = custs[np.argsort(demands[custs])]
            for i in custs:
                if load[j] <= capacities[j] + 1e-9:
                    break
                # Find facility with most remaining capacity
                rem = capacities - load
                rem[j] = -np.inf
                feasible = np.where(rem >= demands[i])[0]
                if len(feasible) > 0:
                    f = int(feasible[np.argmin(rem[feasible])])
                else:
                    f = int(np.argmax(rem))
                result[i] = f
                load[j] -= demands[i]
                load[f] += demands[i]
    
    return result

def _compute_true_cost(demands, capacities, open_cost, assign_cost, result):
    """Compute true objective value."""
    n_fac = len(capacities)
    total = 0.0
    for i in range(len(demands)):
        total += assign_cost[i, result[i]]
    counts = np.bincount(result, minlength=n_fac)
    for j in range(n_fac):
        if counts[j] > 0:
            total += open_cost[j]
    return total

def _greedy_init(demands, capacities, open_cost, assign_cost, seed=0):
    """Greedy initialization with randomization for diversity."""
    rng = np.random.RandomState(seed)
    n_cust, n_fac = len(demands), len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    
    # Sort by demand descending with randomization
    order = np.argsort(-demands + rng.uniform(0, 5, n_cust))
    
    for i in order:
        feasible = np.where(load + demands[i] <= capacities)[0]
        if len(feasible) == 0:
            f = int(np.argmax(capacities - load))
        else:
            # Effective cost: assignment cost + amortized open cost
            eff = assign_cost[i, feasible].copy()
            for idx, f in enumerate(feasible):
                if load[f] == 0:
                    # Amortize open cost over expected customers
                    # Use a heuristic: open_cost / (estimated customers)
                    eff[idx] += open_cost[f] * rng.uniform(0.5, 1.0)
            f = int(feasible[np.argmin(eff)])
        result[i] = f
        load[f] += demands[i]
    
    return result

def _vectorized_local_search(demands, capacities, open_cost, assign_cost, result, 
                              time_limit=20.0):
    """Fast local search: vectorized delta computation per round."""
    n_cust, n_fac = len(demands), len(capacities)
    start_time = time.time()
    result = result.copy()
    
    load = np.zeros(n_fac)
    counts = np.zeros(n_fac)
    np.add.at(load, result, demands)
    np.add.at(counts, result, 1)
    
    current_cost = _compute_true_cost(demands, capacities, open_cost, assign_cost, result)
    best_cost = current_cost
    best_result = result.copy()
    
    no_improve_rounds = 0
    max_no_improve = 10
    
    while time.time() - start_time < time_limit:
        if no_improve_rounds >= max_no_improve:
            break
        
        improved = False
        
        # Process in random order
        order = np.random.permutation(n_cust)
        
        for i in order:
            if time.time() - start_time > time_limit:
                break
            
            cur_f = result[i]
            d_i = demands[i]
            
            # Delta for moving i to each facility
            delta = assign_cost[i, :] - assign_cost[i, cur_f]
            
            # Open cost adjustments
            if counts[cur_f] == 1:
                delta -= open_cost[cur_f]
            for f in range(n_fac):
                if f != cur_f and counts[f] == 0:
                    delta[f] += open_cost[f]
            
            # Capacity constraint
            delta[cur_f] = np.inf
            infeasible = load + d_i > capacities
            delta[infeasible] = np.inf
            
            best_f = int(np.argmin(delta))
            if delta[best_f] < -1e-9:
                old_f = cur_f
                result[i] = best_f
                load[old_f] -= d_i
                load[best_f] += d_i
                counts[old_f] -= 1
                counts[best_f] += 1
                current_cost += delta[best_f]
                improved = True
        
        if improved:
            no_improve_rounds = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_result = result.copy()
        else:
            no_improve_rounds += 1
    
    if best_cost < current_cost - 1e-9:
        return best_result, best_cost
    return result, current_cost

def _pair_exchange_search(demands, capacities, open_cost, assign_cost, result,
                          time_limit=10.0):
    """Pair-exchange local search: swap two customers between facilities."""
    n_cust, n_fac = len(demands), len(capacities)
    start_time = time.time()
    result = result.copy()
    
    load = np.zeros(n_fac)
    counts = np.zeros(n_fac)
    np.add.at(load, result, demands)
    np.add.at(counts, result, 1)
    
    current_cost = _compute_true_cost(demands, capacities, open_cost, assign_cost, result)
    
    # Build customer lists per facility
    fac_custs = [[] for _ in range(n_fac)]
    for i in range(n_cust):
        fac_custs[result[i]].append(i)
    
    improved_any = True
    max_rounds = 50
    round_num = 0
    
    while improved_any and time.time() - start_time < time_limit and round_num < max_rounds:
        round_num += 1
        improved_any = False
        
        # Sample random pairs of customers from different facilities
        n_pairs = min(2000, n_cust * (n_cust - 1) // 2)
        for _ in range(n_pairs):
            if time.time() - start_time > time_limit:
                break
            
            i = np.random.randint(n_cust)
            j = np.random.randint(n_cust)
            if i == j:
                continue
            
            fi = result[i]
            fj = result[j]
            if fi == fj:
                continue
            
            # Check capacity feasibility
            if load[fi] - demands[i] + demands[j] > capacities[fi] + 1e-9:
                continue
            if load[fj] - demands[j] + demands[i] > capacities[fj] + 1e-9:
                continue
            
            # Compute delta
            delta = (assign_cost[i, fj] - assign_cost[i, fi]) + \
                    (assign_cost[j, fi] - assign_cost[j, fj])
            
            # Open cost changes (counts don't change since both facilities stay open
            # unless one had exactly 1 customer)
            if counts[fi] == 1:
                # fi loses i but gains j, so count stays 1
                pass
            if counts[fj] == 1:
                # fj loses j but gains i, so count stays 1
                pass
            
            if delta < -1e-9:
                # Apply swap
                result[i] = fj
                result[j] = fi
                load[fi] += demands[j] - demands[i]
                load[fj] += demands[i] - demands[j]
                # Counts unchanged
                current_cost += delta
                fac_custs[fi].remove(i)
                fac_custs[fi].append(j)
                fac_custs[fj].remove(j)
                fac_custs[fj].append(i)
                improved_any = True
    
    return result, current_cost

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust, n_fac = len(demands), len(capacities)
    
    best_result = None
    best_cost = np.inf
    
    # Phase 1: Lagrangian relaxation (25s)
    lag_time = min(25.0, deadline - time.time() - 10.0)
    if lag_time > 2.0:
        result, cost, dual = _subgradient_lagrangian(
            demands, capacities, open_cost, assign_cost,
            max_iter=150, time_limit=lag_time)
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Ensure we have a feasible starting point
    if best_result is None:
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost, seed=42)
        best_cost = _compute_true_cost(demands, capacities, open_cost, assign_cost, best_result)
    
    # Phase 2: Local search with multiple restarts
    remaining = deadline - time.time() - 5.0
    
    # First pass: local search on the Lagrangian/greedy solution
    if remaining > 5.0:
        ls_time = min(30.0, remaining * 0.5)
        result, cost = _vectorized_local_search(
            demands, capacities, open_cost, assign_cost, best_result,
            time_limit=ls_time)
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Pair exchange polish
    remaining = deadline - time.time() - 5.0
    if remaining > 3.0:
        pe_time = min(10.0, remaining * 0.4)
        result, cost = _pair_exchange_search(
            demands, capacities, open_cost, assign_cost, best_result,
            time_limit=pe_time)
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Phase 3: Additional restarts with diverse initializations
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        n_restarts = max(1, int(remaining / 20.0))
        for r in range(n_restarts):
            if time.time() > deadline - 8.0:
                break
            
            start = _greedy_init(demands, capacities, open_cost, assign_cost, 
                                seed=r * 137 + 7 + 1000)
            
            time_left = deadline - time.time() - 8.0
            if time_left < 5.0:
                break
            
            ls_time = min(15.0, time_left * 0.6)
            result, cost = _vectorized_local_search(
                demands, capacities, open_cost, assign_cost, start,
                time_limit=ls_time)
            
            if cost < best_cost:
                best_cost = cost
                best_result = result.copy()
            
            # Quick pair exchange on the new best
            time_left = deadline - time.time() - 8.0
            if time_left > 3.0:
                pe_time = min(5.0, time_left * 0.5)
                result, cost = _pair_exchange_search(
                    demands, capacities, open_cost, assign_cost, best_result,
                    time_limit=pe_time)
                if cost < best_cost:
                    best_cost = cost
                    best_result = result.copy()
    
    # Final local search polish on the best solution
    remaining = deadline - time.time() - 3.0
    if remaining > 3.0 and best_result is not None:
        result, cost = _vectorized_local_search(
            demands, capacities, open_cost, assign_cost, best_result,
            time_limit=min(10.0, remaining))
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Final feasibility check and repair
    if best_result is None:
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost, seed=0)
    
    load = np.zeros(n_fac)
    np.add.at(load, best_result, demands)
    
    for j in range(n_fac):
        if load[j] > capacities[j] + 1e-9:
            custs = np.where(best_result == j)[0]
            custs = custs[np.argsort(demands[custs])]
            for i in custs:
                if load[j] <= capacities[j] + 1e-9:
                    break
                rem = capacities - load
                rem[j] = -np.inf
                feasible = np.where(rem >= demands[i])[0]
                if len(feasible) > 0:
                    f = int(feasible[np.argmin(rem[feasible])])
                else:
                    f = int(np.argmax(rem))
                best_result[i] = f
                load[j] -= demands[i]
                load[f] += demands[i]
    
    best_result = best_result.astype(int)
    for i in range(n_cust):
        if best_result[i] < 0 or best_result[i] >= n_fac:
            best_result[i] = 0
    
    return best_result
# EVOLVE-BLOCK-END
