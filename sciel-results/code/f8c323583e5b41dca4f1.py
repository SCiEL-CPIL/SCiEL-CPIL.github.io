"""Capacitated facility location: SEARCH approach with diverse candidates + local refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using a search approach:
    generate many diverse initial candidates, refine each with local search,
    and keep the best valid solution found within the time budget.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    # Deadline with safety margin: evaluator kills at 150s, aim to finish by ~145s
    deadline = time.time() + 145.0
    
    def calc_cost(res):
        """Compute total cost: open cost + assignment cost."""
        used = np.unique(res)
        open_sum = float(np.sum(open_cost[used]))
        assign_sum = float(np.sum(assign_cost[np.arange(n_cust), res]))
        return open_sum + assign_sum
    
    def is_valid(res, load):
        """Check if solution is valid (no capacity violations, indices in range)."""
        if np.any(res < 0) or np.any(res >= n_fac):
            return False
        if np.any(load > capacities):
            return False
        if np.any(load < 0):
            return False
        return True
    
    def loads_of(res):
        load = np.zeros(n_fac, dtype=np.int64)
        np.add.at(load, res, demands)
        return load
    
    # ---- Local search: single moves, swaps, and close-facility ----
    def local_search(res, load, max_time=10.0):
        """
        Refine solution with local search moves.
        Returns refined (res, load, score).
        """
        res = res.copy()
        load = load.copy()
        start_time = time.time()
        
        improved = True
        while improved and (time.time() - start_time) < max_time and time.time() < deadline - 2.0:
            improved = False
            
            # --- Single move best-improvement pass ---
            any_improve = False
            for i in range(n_cust):
                if time.time() > deadline - 2.0:
                    break
                f_old = res[i]
                d_i = demands[i]
                
                # Find feasible target facilities
                feasible = np.where((load + d_i <= capacities) & (np.arange(n_fac) != f_old))[0]
                if len(feasible) == 0:
                    continue
                
                # Compute delta for each feasible target
                base_assign = assign_cost[i, f_old]
                deltas = assign_cost[i, feasible] - base_assign
                
                # Open cost adjustment
                if load[f_old] == d_i:
                    # f_old becomes empty: save open_cost[f_old]
                    deltas = deltas - open_cost[f_old]
                
                # Opening new facility penalty
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
                # Try swap moves even if no single move improved
                pass
            
            # --- Swap moves (sampled more aggressively) ---
            rng = np.random.default_rng(hash(tuple(res)) % 2**32)
            n_swap_samples = 1000
            for _ in range(n_swap_samples):
                if time.time() > deadline - 2.0:
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
                
                # Check capacity after swap
                if load[f_j] - d_j + d_i > capacities[f_j]:
                    continue
                if load[f_i] - d_i + d_j > capacities[f_i]:
                    continue
                
                # Compute delta
                delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                               (assign_cost[j, f_i] - assign_cost[j, f_j])
                
                # Open cost: check if any facility becomes empty or opens
                # f_i: load changes from L_i to L_i - d_i + d_j
                # f_j: load changes from L_j to L_j - d_j + d_i
                old_open_i = open_cost[f_i] if load[f_i] > 0 else 0.0
                old_open_j = open_cost[f_j] if load[f_j] > 0 else 0.0
                new_load_i = load[f_i] - d_i + d_j
                new_load_j = load[f_j] - d_j + d_i
                new_open_i = open_cost[f_i] if new_load_i > 0 else 0.0
                new_open_j = open_cost[f_j] if new_load_j > 0 else 0.0
                delta_open = (new_open_i + new_open_j) - (old_open_i + old_open_j)
                
                total_delta = delta_assign + delta_open
                if total_delta < -1e-9:
                    res[i], res[j] = res[j], res[i]
                    load[f_i] = new_load_i
                    load[f_j] = new_load_j
                    improved = True
            
            # --- Close-facility move (improved: try multiple strategies) ---
            for f_close in range(n_fac):
                if time.time() > deadline - 2.0:
                    break
                if load[f_close] == 0:
                    continue
                
                custs = np.where(res == f_close)[0]
                if len(custs) == 0:
                    continue
                
                # Try to reassign all customers of f_close to other facilities
                # Strategy 1: Sort by demand descending
                temp_load = load.copy()
                temp_res = res.copy()
                temp_load[f_close] = 0
                
                success = True
                order = custs[np.argsort(-demands[custs])]
                for c in order:
                    d_c = demands[c]
                    feasible = np.where(temp_load + d_c <= capacities)[0]
                    if len(feasible) == 0:
                        success = False
                        break
                    # Choose facility minimizing assign_cost + open penalty
                    costs = assign_cost[c, feasible]
                    open_pen = np.where(temp_load[feasible] == 0, open_cost[feasible], 0.0)
                    adjusted = costs + open_pen
                    best_f = int(feasible[np.argmin(adjusted)])
                    temp_res[c] = best_f
                    temp_load[best_f] += d_c
                
                if success:
                    # Compute delta
                    old_assign_sum = float(np.sum(assign_cost[custs, f_close]))
                    new_assign_sum = float(np.sum(assign_cost[custs, temp_res[custs]]))
                    old_open = float(np.sum(open_cost[load > 0]))
                    new_open = float(np.sum(open_cost[temp_load > 0]))
                    delta = (new_assign_sum - old_assign_sum) + (new_open - old_open)
                    
                    if delta < -1e-9:
                        res[:] = temp_res
                        load[:] = temp_load
                        improved = True
                
                # Strategy 2: Sort by assignment cost to f_close ascending (cheap to move first)
                temp_load2 = load.copy()
                temp_res2 = res.copy()
                temp_load2[f_close] = 0
                
                success2 = True
                order2 = custs[np.argsort(assign_cost[custs, f_close])]
                for c in order2:
                    d_c = demands[c]
                    feasible = np.where(temp_load2 + d_c <= capacities)[0]
                    if len(feasible) == 0:
                        success2 = False
                        break
                    costs = assign_cost[c, feasible]
                    open_pen = np.where(temp_load2[feasible] == 0, open_cost[feasible], 0.0)
                    adjusted = costs + open_pen
                    best_f = int(feasible[np.argmin(adjusted)])
                    temp_res2[c] = best_f
                    temp_load2[best_f] += d_c
                
                if success2:
                    old_assign_sum = float(np.sum(assign_cost[custs, f_close]))
                    new_assign_sum = float(np.sum(assign_cost[custs, temp_res2[custs]]))
                    old_open = float(np.sum(open_cost[load > 0]))
                    new_open = float(np.sum(open_cost[temp_load2 > 0]))
                    delta = (new_assign_sum - old_assign_sum) + (new_open - old_open)
                    
                    if delta < -1e-9:
                        res[:] = temp_res2
                        load[:] = temp_load2
                        improved = True
        
        score = calc_cost(res)
        return res, load, score
    
    # ---- Generate diverse candidate solutions ----
    def generate_candidate(method, rng):
        """
        Generate an initial feasible assignment using the given method.
        Returns (res, load) or None if infeasible.
        """
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac, dtype=np.int64)
        
        if method == 0:
            # Greedy by demand descending, cheapest feasible
            order = np.argsort(-demands)
            for i in order:
                d_i = demands[i]
                feasible = np.where(load + d_i <= capacities)[0]
                if len(feasible) == 0:
                    return None
                costs = assign_cost[i, feasible]
                f = int(feasible[np.argmin(costs)])
                res[i] = f
                load[f] += d_i
        
        elif method == 1:
            # Greedy by demand descending, with random tie-breaking
            order = np.argsort(-demands)
            for i in order:
                d_i = demands[i]
                feasible = np.where(load + d_i <= capacities)[0]
                if len(feasible) == 0:
                    return None
                if len(feasible) > 1:
                    # Randomly pick among top-3 cheapest
                    costs = assign_cost[i, feasible]
                    top_k = min(3, len(feasible))
                    top_idx = np.argsort(costs)[:top_k]
                    f = int(feasible[top_idx[rng.integers(top_k)]])
                else:
                    f = int(feasible[0])
                res[i] = f
                load[f] += d_i
        
        elif method == 2:
            # Random order, cheapest feasible
            order = rng.permutation(n_cust)
            for i in order:
                d_i = demands[i]
                feasible = np.where(load + d_i <= capacities)[0]
                if len(feasible) == 0:
                    return None
                costs = assign_cost[i, feasible]
                f = int(feasible[np.argmin(costs)])
                res[i] = f
                load[f] += d_i
        
        elif method == 3:
            # Greedy with open cost penalty (Lagrangian-like)
            alpha = 50.0  # penalty weight
            eff_cost = assign_cost + alpha * np.outer(np.ones(n_cust), open_cost / np.sum(open_cost))
            order = np.argsort(-demands)
            for i in order:
                d_i = demands[i]
                feasible = np.where(load + d_i <= capacities)[0]
                if len(feasible) == 0:
                    return None
                costs = eff_cost[i, feasible]
                f = int(feasible[np.argmin(costs)])
                res[i] = f
                load[f] += d_i
        
        elif method == 4:
            # Greedy by assignment cost ascending (cheap first)
            order = np.argsort(np.min(assign_cost, axis=1))
            for i in order:
                d_i = demands[i]
                feasible = np.where(load + d_i <= capacities)[0]
                if len(feasible) == 0:
                    return None
                costs = assign_cost[i, feasible]
                f = int(feasible[np.argmin(costs)])
                res[i] = f
                load[f] += d_i
        
        elif method == 5:
            # Random feasible assignment
            order = rng.permutation(n_cust)
            for i in order:
                d_i = demands[i]
                feasible = np.where(load + d_i <= capacities)[0]
                if len(feasible) == 0:
                    return None
                if len(feasible) > 1:
                    f = int(feasible[rng.integers(len(feasible))])
                else:
                    f = int(feasible[0])
                res[i] = f
                load[f] += d_i
        
        elif method == 6:
            # Greedy with capacity balancing: prefer less-loaded facilities
            order = np.argsort(-demands)
            for i in order:
                d_i = demands[i]
                feasible = np.where(load + d_i <= capacities)[0]
                if len(feasible) == 0:
                    return None
                # Score: assign_cost + 0.1 * (load/facility_capacity)
                frac = load[feasible] / (capacities[feasible] + 1e-9)
                scores = assign_cost[i, feasible] + 0.1 * frac * 100
                f = int(feasible[np.argmin(scores)])
                res[i] = f
                load[f] += d_i
        
        elif method == 7:
            # Minimize number of open facilities first, then assign cost
            # Assign all to cheapest, but try to consolidate
            order = np.argsort(-demands)
            for i in order:
                d_i = demands[i]
                feasible = np.where(load + d_i <= capacities)[0]
                if len(feasible) == 0:
                    return None
                # Prefer already-open facilities with low assign cost
                open_fac = feasible[load[feasible] > 0]
                if len(open_fac) > 0:
                    costs = assign_cost[i, open_fac]
                    f = int(open_fac[np.argmin(costs)])
                else:
                    costs = assign_cost[i, feasible]
                    f = int(feasible[np.argmin(costs)])
                res[i] = f
                load[f] += d_i
        
        elif method == 8:
            # Randomized greedy with multiple alpha values
            alpha = rng.uniform(0, 20)
            eff_cost = assign_cost + alpha * np.outer(np.ones(n_cust), open_cost / np.sum(open_cost))
            order = np.argsort(-demands)
            for i in order:
                d_i = demands[i]
                feasible = np.where(load + d_i <= capacities)[0]
                if len(feasible) == 0:
                    return None
                costs = eff_cost[i, feasible]
                # Add small random perturbation
                costs += rng.uniform(0, 5, size=len(costs))
                f = int(feasible[np.argmin(costs)])
                res[i] = f
                load[f] += d_i
        
        elif method == 9:
            # Two-phase: first assign to minimize opens, then refine
            # Phase 1: assign to reduce open facilities
            order = np.argsort(-demands)
            for i in order:
                d_i = demands[i]
                feasible = np.where(load + d_i <= capacities)[0]
                if len(feasible) == 0:
                    return None
                # Prefer facilities with highest remaining capacity (to consolidate)
                rem_cap = capacities[feasible] - load[feasible]
                f = int(feasible[np.argmax(rem_cap)])
                res[i] = f
                load[f] += d_i
        
        else:
            # Fallback: greedy
            order = np.argsort(-demands)
            for i in order:
                d_i = demands[i]
                feasible = np.where(load + d_i <= capacities)[0]
                if len(feasible) == 0:
                    return None
                costs = assign_cost[i, feasible]
                f = int(feasible[np.argmin(costs)])
                res[i] = f
                load[f] += d_i
        
        # Validate
        if not is_valid(res, load):
            return None
        
        return res, load
    
    # ---- Main search loop ----
    rng = np.random.default_rng(42)
    
    best_result = None
    best_score = np.inf
    
    # Generate and evaluate many candidates
    n_candidates = 200
    methods = list(range(10))
    
    for k in range(n_candidates):
        if time.time() > deadline - 20.0:
            break
        
        method = methods[k % len(methods)]
        candidate = generate_candidate(method, rng)
        
        if candidate is None:
            continue
        
        res, load = candidate
        
        # Compute initial score
        score = calc_cost(res)
        
        # Refine with local search: more time for early candidates
        remaining = deadline - time.time()
        if k < 20:
            ls_time = min(8.0, remaining * 0.03)
        elif k < 60:
            ls_time = min(5.0, remaining * 0.02)
        else:
            ls_time = min(3.0, remaining * 0.015)
        if ls_time < 0.3:
            ls_time = 0.3
        
        res, load, score = local_search(res, load, max_time=ls_time)
        
        if is_valid(res, load) and score < best_score:
            best_score = score
            best_result = res.copy()
    
    # If no valid result found, use simple greedy as fallback
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
    
    # Final high-precision polish pass on the best solution
    if time.time() < deadline - 5.0:
        final_load = loads_of(best_result)
        polish_time = min(10.0, deadline - time.time() - 3.0)
        if polish_time > 1.0:
            best_result, final_load, best_score = local_search(best_result, final_load, max_time=polish_time)
    
    # Final validation
    final_load = loads_of(best_result)
    if not is_valid(best_result, final_load):
        # Emergency fallback
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
# EVOLVE-BLOCK-END
