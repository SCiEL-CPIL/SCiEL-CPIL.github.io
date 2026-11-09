"""Capacitated facility location: Lagrangian relaxation with vectorized local search."""
import time
import numpy as np


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns each customer to a facility using Lagrangian relaxation for initialization
    followed by strong vectorized local search.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    deadline = time.time() + 145.0

    def calc_cost(res):
        used = np.unique(res)
        if len(used) == 0:
            return 0.0
        open_sum = float(np.sum(open_cost[used]))
        assign_sum = float(np.sum(assign_cost[np.arange(n_cust), res]))
        return open_sum + assign_sum

    def loads_of(res):
        load = np.zeros(n_fac, dtype=np.int64)
        np.add.at(load, res, demands)
        return load

    def is_valid(res, load):
        if res is None or len(res) != n_cust:
            return False
        if np.any(res < 0) or np.any(res >= n_fac):
            return False
        if np.any(load > capacities) or np.any(load < 0):
            return False
        return True

    def repair(res):
        """Repair an infeasible assignment by moving customers from overloaded facilities."""
        res = res.copy()
        load = loads_of(res)
        
        overloaded = np.where(load > capacities)[0]
        for f_over in overloaded:
            custs = np.where(res == f_over)[0]
            custs = custs[np.argsort(-demands[custs])]
            
            for c in custs:
                if load[f_over] <= capacities[f_over]:
                    break
                d_c = demands[c]
                feasible = np.where(load + d_c <= capacities)[0]
                feasible = feasible[feasible != f_over]
                if len(feasible) == 0:
                    feasible = np.where(load + d_c <= capacities)[0]
                if len(feasible) == 0:
                    continue
                costs = assign_cost[c, feasible]
                best_f = feasible[np.argmin(costs)]
                res[c] = best_f
                load[f_over] -= d_c
                load[best_f] += d_c
        return res, load

    def local_search(res, load, max_time=10.0):
        """
        Refine solution with vectorized single moves, swaps, and facility closing.
        """
        res = res.copy()
        load = load.copy()
        start_time = time.time()
        
        improved = True
        while improved and (time.time() - start_time) < max_time and time.time() < deadline - 3.0:
            improved = False
            
            # --- Vectorized Single Move Pass ---
            any_improve = False
            for i in range(n_cust):
                if time.time() > deadline - 3.0:
                    break
                f_old = res[i]
                d_i = demands[i]
                
                feasible_mask = (load + d_i <= capacities)
                feasible_mask[f_old] = False
                feasible = np.where(feasible_mask)[0]
                
                if len(feasible) == 0:
                    continue
                
                base_assign = assign_cost[i, f_old]
                deltas = assign_cost[i, feasible] - base_assign
                
                if load[f_old] == d_i:
                    deltas = deltas - open_cost[f_old]
                
                for idx, f_new in enumerate(feasible):
                    if load[f_new] == 0:
                        deltas[idx] += open_cost[f_new]
                
                best_idx = int(np.argmin(deltas))
                best_delta = deltas[best_idx]
                
                if best_delta < -1e-9:
                    f_new = feasible[best_idx]
                    res[i] = f_new
                    load[f_old] -= d_i
                    load[f_new] += d_i
                    any_improve = True
                    improved = True
            
            if not any_improve:
                # --- Swap Moves ---
                rng = np.random.default_rng(int(time.time() * 10000) % (2**32))
                n_swap_samples = 2000
                for _ in range(n_swap_samples):
                    if time.time() > deadline - 3.0:
                        break
                    i, j = rng.integers(0, n_cust, size=2)
                    if i == j:
                        continue
                    f_i = res[i]
                    f_j = res[j]
                    if f_i == f_j:
                        continue
                    d_i = demands[i]
                    d_j = demands[j]
                    
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    
                    delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                   (assign_cost[j, f_i] - assign_cost[j, f_j])
                    
                    old_load_i = load[f_i]
                    old_load_j = load[f_j]
                    new_load_i = old_load_i - d_i + d_j
                    new_load_j = old_load_j - d_j + d_i
                    
                    old_open_i = open_cost[f_i] if old_load_i > 0 else 0.0
                    old_open_j = open_cost[f_j] if old_load_j > 0 else 0.0
                    new_open_i = open_cost[f_i] if new_load_i > 0 else 0.0
                    new_open_j = open_cost[f_j] if new_load_j > 0 else 0.0
                    delta_open = (new_open_i + new_open_j) - (old_open_i + old_open_j)
                    
                    total_delta = delta_assign + delta_open
                    if total_delta < -1e-9:
                        res[i], res[j] = res[j], res[i]
                        load[f_i] = new_load_i
                        load[f_j] = new_load_j
                        improved = True
                
                # --- Close Facility Move ---
                for f_close in range(n_fac):
                    if time.time() > deadline - 3.0:
                        break
                    if load[f_close] == 0:
                        continue
                    
                    custs = np.where(res == f_close)[0]
                    if len(custs) == 0:
                        continue
                    
                    temp_load = load.copy()
                    temp_res = res.copy()
                    temp_load[f_close] = 0
                    
                    success = True
                    order = custs[np.argsort(-demands[custs])]
                    
                    for c in order:
                        d_c = demands[c]
                        feasible_mask = (temp_load + d_c <= capacities)
                        feasible_mask[f_close] = False
                        feasible = np.where(feasible_mask)[0]
                        
                        if len(feasible) == 0:
                            success = False
                            break
                        
                        costs = assign_cost[c, feasible]
                        open_pen = np.where(temp_load[feasible] == 0, open_cost[feasible], 0.0)
                        adjusted = costs + open_pen
                        best_f = int(feasible[np.argmin(adjusted)])
                        temp_res[c] = best_f
                        temp_load[best_f] += d_c
                    
                    if success:
                        old_assign_sum = float(np.sum(assign_cost[custs, f_close]))
                        new_assign_sum = float(np.sum(assign_cost[custs, temp_res[custs]]))
                        
                        old_open = float(np.sum(open_cost[load > 0]))
                        new_open = float(np.sum(open_cost[temp_load > 0]))
                        
                        delta = (new_assign_sum - old_assign_sum) + (new_open - old_open)
                        
                        if delta < -1e-9:
                            res[:] = temp_res
                            load[:] = temp_load
                            improved = True
        
        return res, load

    # --- Lagrangian Relaxation Initialization ---
    def lagrangian_init(num_iters=100, step_size=5.0):
        """
        Lagrangian relaxation: dualize capacity constraints, solve subproblems,
        update multipliers via subgradient method.
        """
        dual_vars = np.zeros(n_fac)
        best_sol = None
        best_cost = np.inf
        
        for it in range(num_iters):
            if time.time() > deadline - 15.0:
                break
            
            # Solve subproblem: for each customer, pick facility minimizing
            # assign_cost[i,f] + dual_vars[f] * demands[i]
            adjusted_costs = assign_cost + np.outer(demands, dual_vars)
            best_facs = np.argmin(adjusted_costs, axis=1)
            
            # Check capacity violations
            load = np.zeros(n_fac, dtype=np.int64)
            np.add.at(load, best_facs, demands)
            
            # Compute subgradient
            violations = load.astype(np.float64) - capacities.astype(np.float64)
            
            # Update dual variables (subgradient method)
            # Use diminishing step size
            effective_step = step_size / (1.0 + it * 0.01)
            dual_vars = np.maximum(0, dual_vars + effective_step * violations / 100.0)
            
            # Repair: move customers from overloaded facilities
            res = best_facs.copy()
            temp_load = load.copy()
            
            overloaded = np.where(temp_load > capacities)[0]
            
            for f_over in overloaded:
                custs = np.where(res == f_over)[0]
                custs = custs[np.argsort(-demands[custs])]
                
                for c in custs:
                    if temp_load[f_over] <= capacities[f_over]:
                        break
                    
                    d_c = demands[c]
                    feasible = np.where(temp_load + d_c <= capacities)[0]
                    feasible = feasible[feasible != f_over]
                    
                    if len(feasible) == 0:
                        feasible = np.where(temp_load + d_c <= capacities)[0]
                    
                    if len(feasible) == 0:
                        continue
                    
                    costs = assign_cost[c, feasible]
                    best_f = feasible[np.argmin(costs)]
                    
                    res[c] = best_f
                    temp_load[f_over] -= d_c
                    temp_load[best_f] += d_c
            
            if is_valid(res, temp_load):
                cost = calc_cost(res)
                if cost < best_cost:
                    best_cost = cost
                    best_sol = res.copy()
            
            # Early stop if converged
            if np.max(np.abs(violations)) < 1:
                break
        
        return best_sol, best_cost

    # --- Generate diverse initial solutions ---
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # Strategy 1: Lagrangian with different step sizes
    if time.time() < deadline - 20.0:
        lag_res, lag_cost = lagrangian_init(num_iters=100, step_size=5.0)
        if lag_res is not None:
            lag_load = loads_of(lag_res)
            if is_valid(lag_res, lag_load):
                candidates.append((lag_res, lag_load))
    
    if time.time() < deadline - 15.0:
        lag_res, lag_cost = lagrangian_init(num_iters=80, step_size=10.0)
        if lag_res is not None:
            lag_load = loads_of(lag_res)
            if is_valid(lag_res, lag_load):
                candidates.append((lag_res, lag_load))
    
    # Strategy 2: Greedy with demand-descending order
    order1 = np.argsort(-demands)
    res = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=np.int64)
    for i in order1:
        d_i = demands[i]
        feasible = np.where(load + d_i <= capacities)[0]
        if len(feasible) == 0:
            continue
        costs = assign_cost[i, feasible]
        f = int(feasible[np.argmin(costs)])
        res[i] = f
        load[f] += d_i
    if is_valid(res, load):
        candidates.append((res, load))
    
    # Strategy 3: Greedy with open cost penalty
    res = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=np.int64)
    for i in order1:
        d_i = demands[i]
        feasible = np.where(load + d_i <= capacities)[0]
        if len(feasible) == 0:
            continue
        scores = assign_cost[i, feasible].copy()
        for idx, f in enumerate(feasible):
            if load[f] == 0:
                scores[idx] += open_cost[f]
        f = int(feasible[np.argmin(scores)])
        res[i] = f
        load[f] += d_i
    if is_valid(res, load):
        candidates.append((res, load))
    
    # Strategy 4: Random order greedy
    order4 = rng.permutation(n_cust)
    res = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=np.int64)
    for i in order4:
        d_i = demands[i]
        feasible = np.where(load + d_i <= capacities)[0]
        if len(feasible) == 0:
            continue
        costs = assign_cost[i, feasible]
        f = int(feasible[np.argmin(costs)])
        res[i] = f
        load[f] += d_i
    if is_valid(res, load):
        candidates.append((res, load))
    
    # Evaluate and refine candidates
    best_result = None
    best_score = np.inf
    
    for idx, (res, load) in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
        
        score = calc_cost(res)
        
        remaining = deadline - time.time()
        if idx < 2:
            ls_time = min(12.0, remaining * 0.12)
        elif idx < 4:
            ls_time = min(6.0, remaining * 0.06)
        else:
            ls_time = min(3.0, remaining * 0.03)
        
        if ls_time < 1.0:
            ls_time = 1.0
        
        res, load = local_search(res, load, max_time=ls_time)
        score = calc_cost(res)
        
        if is_valid(res, load) and score < best_score:
            best_score = score
            best_result = res.copy()
    
    # Fallback if no valid candidate
    if best_result is None:
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac, dtype=np.int64)
        order = np.argsort(-demands)
        for i in order:
            d_i = demands[i]
            feasible = np.where(load + d_i <= capacities)[0]
            if len(feasible) == 0:
                f = int(np.argmax(capacities - load))
            else:
                costs = assign_cost[i, feasible]
                f = int(feasible[np.argmin(costs)])
            res[i] = f
            load[f] += d_i
        best_result = res
    
    # Final polish
    if time.time() < deadline - 5.0:
        final_load = loads_of(best_result)
        polish_time = min(15.0, deadline - time.time() - 3.0)
        if polish_time > 2.0:
            best_result, final_load = local_search(best_result, final_load, max_time=polish_time)
    
    # Final validation
    final_load = loads_of(best_result)
    if not is_valid(best_result, final_load):
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac, dtype=np.int64)
        order = np.argsort(-demands)
        for i in order:
            d_i = demands[i]
            feasible = np.where(load + d_i <= capacities)[0]
            if len(feasible) == 0:
                f = int(np.argmax(capacities - load))
            else:
                costs = assign_cost[i, feasible]
                f = int(feasible[np.argmin(costs)])
            res[i] = f
            load[f] += d_i
        return res.astype(int)
    
    return best_result.astype(int)
