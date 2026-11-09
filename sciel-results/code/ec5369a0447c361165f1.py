import time
import numpy as np
import random

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns each customer to a facility using a strong local search approach.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    # Deadline with safety margin (150s limit, keep 10s buffer)
    deadline = time.time() + 140.0

    def calc_cost(res):
        used = np.unique(res)
        if len(used) == 0:
            return 0.0
        open_sum = float(np.sum(open_cost[used]))
        assign_sum = float(np.sum(assign_cost[np.arange(n_cust), res]))
        return open_sum + assign_sum

    def is_valid(res, load):
        if res is None or len(res) != n_cust:
            return False
        if np.any(res < 0) or np.any(res >= n_fac):
            return False
        if np.any(load > capacities) or np.any(load < 0):
            return False
        return True

    def loads_of(res):
        load = np.zeros(n_fac, dtype=np.int64)
        np.add.at(load, res, demands)
        return load

    def make_feasible(res, load):
        """
        If the assignment violates capacity, reassign excess customers to other facilities.
        """
        res = res.copy()
        load = load.copy()
        
        excess_facs = np.where(load > capacities)[0]
        
        for f_excess in excess_facs:
            while load[f_excess] > capacities[f_excess]:
                custs = np.where(res == f_excess)[0]
                if len(custs) == 0:
                    break
                    
                min_delta = np.inf
                best_c = -1
                best_f_new = -1
                
                for c in custs:
                    d_c = demands[c]
                    if d_c > capacities.max():
                        continue
                    
                    feasible = np.where(load + d_c <= capacities)[0]
                    if len(feasible) == 0:
                        break
                    
                    feasible = feasible[feasible != f_excess]
                    if len(feasible) == 0:
                        break
                    
                    old_load_f = load[f_excess]
                    new_load_f = old_load_f - d_c
                    
                    for f_new in feasible:
                        new_load_f_new = load[f_new] + d_c
                        
                        delta_assign = assign_cost[c, f_new] - assign_cost[c, f_excess]
                        
                        old_open_f = open_cost[f_excess] if old_load_f > 0 else 0.0
                        new_open_f = open_cost[f_excess] if new_load_f > 0 else 0.0
                        old_open_f_new = open_cost[f_new] if load[f_new] > 0 else 0.0
                        new_open_f_new = open_cost[f_new] if new_load_f_new > 0 else 0.0
                        
                        delta_open = (new_open_f + new_open_f_new) - (old_open_f + old_open_f_new)
                        
                        total_delta = delta_assign + delta_open
                        
                        if total_delta < min_delta:
                            min_delta = total_delta
                            best_c = c
                            best_f_new = f_new
                
                if best_c == -1:
                    break
                
                res[best_c] = best_f_new
                load[f_excess] -= demands[best_c]
                load[best_f_new] += demands[best_c]
                
        return res, load

    def greedy_construct(order, open_penalty_scale=1.0):
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac, dtype=np.int64)
        for i in order:
            d_i = demands[i]
            feasible = np.where(load + d_i <= capacities)[0]
            if len(feasible) == 0:
                return None
            scores = assign_cost[i, feasible].copy()
            for idx, f in enumerate(feasible):
                if load[f] == 0:
                    scores[idx] += open_cost[f] * open_penalty_scale
            f = int(feasible[np.argmin(scores)])
            res[i] = f
            load[f] += d_i
        if not is_valid(res, load):
            return None
        return res, load

    def local_search(res, load, max_time=5.0):
        """
        Strong local search with single moves and swaps.
        """
        res = res.copy()
        load = load.copy()
        start_time = time.time()
        current_score = calc_cost(res)
        best_score = current_score
        best_res = res.copy()
        best_load = load.copy()
        
        # Precompute some arrays for speed
        # We'll use a simple loop-based approach for clarity and correctness
        
        improved = True
        no_improve_count = 0
        max_no_improve = 50
        
        while improved and (time.time() - start_time) < max_time and time.time() < deadline - 5.0:
            improved = False
            
            # Single Move Pass
            for i in range(n_cust):
                if time.time() > deadline - 5.0:
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
                    current_score += best_delta
                    improved = True
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_res = res.copy()
                        best_load = load.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        
            if not improved:
                # Try swaps
                rng = np.random.default_rng(int(time.time() * 10000) % (2**32))
                n_swap_samples = 200
                any_swap_improve = False
                for _ in range(n_swap_samples):
                    if time.time() > deadline - 5.0:
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
                        current_score += total_delta
                        improved = True
                        any_swap_improve = True
                        
                        if current_score < best_score:
                            best_score = current_score
                            best_res = res.copy()
                            best_load = load.copy()
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                
                if not any_swap_improve:
                    # Random restart within local search
                    if no_improve_count > max_no_improve:
                        # Perturb
                        for _ in range(5):
                            i = rng.integers(0, n_cust)
                            f_old = res[i]
                            d_i = demands[i]
                            feasible = np.where(load + d_i <= capacities)[0]
                            feasible = feasible[feasible != f_old]
                            if len(feasible) > 0:
                                f_new = feasible[rng.integers(0, len(feasible))]
                                res[i] = f_new
                                load[f_old] -= d_i
                                load[f_new] += d_i
                                current_score = calc_cost(res)
                        no_improve_count = 0
                    else:
                        break
                        
        return best_res, best_load

    # 1. Generate multiple initial solutions
    candidates = []
    
    # Greedy 1: Sort by demand descending
    order1 = np.argsort(-demands)
    cand = greedy_construct(order1, 1.0)
    if cand: candidates.append(cand)
    
    # Greedy 2: Sort by demand descending, different open penalty
    cand = greedy_construct(order1, 0.5)
    if cand: candidates.append(cand)
    
    # Greedy 3: Random order
    rng = np.random.default_rng(42)
    order3 = rng.permutation(n_cust)
    cand = greedy_construct(order3, 0.8)
    if cand: candidates.append(cand)
    
    # Greedy 4: Sort by assignment cost (to nearest facility)
    min_costs = np.min(assign_cost, axis=1)
    order4 = np.argsort(min_costs)
    cand = greedy_construct(order4, 1.0)
    if cand: candidates.append(cand)
    
    # 2. Refine each candidate with local search
    best_result = None
    best_score = np.inf
    
    # Allocate time based on number of candidates
    n_cands = len(candidates)
    if n_cands == 0:
        # Fallback
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
        return best_result.astype(int)
    
    time_per_candidate = min(20.0, (deadline - time.time() - 10.0) / n_cands)
    
    for idx, (res, load) in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
            
        # First candidate gets more time
        if idx == 0:
            ls_time = min(30.0, deadline - time.time() - 10.0)
        else:
            ls_time = min(15.0, deadline - time.time() - 10.0)
            
        if ls_time < 1.0:
            ls_time = 1.0
            
        res_ls, load_ls = local_search(res, load, max_time=ls_time)
        score = calc_cost(res_ls)
        
        if is_valid(res_ls, load_ls) and score < best_score:
            best_score = score
            best_result = res_ls.copy()
            
    # Fallback
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
        polish_time = min(10.0, deadline - time.time() - 3.0)
        if polish_time > 1.0:
            best_result, final_load = local_search(best_result, final_load, max_time=polish_time)
            
    # Final validation
    final_load = loads_of(best_result)
    if not is_valid(best_result, final_load):
        best_result, final_load = make_feasible(best_result, final_load)
        
    return best_result.astype(int)
