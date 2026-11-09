"""Capacitated facility location: Multi-start heuristic with min-cost flow inspired initialization and strong local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using a multi-start approach with diverse initializations
    and a powerful local search involving moves, swaps, and facility closing.
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
        if res is None:
            return np.inf
        used = np.unique(res)
        if len(used) == 0:
            return 0.0
        open_sum = float(np.sum(open_cost[used]))
        assign_sum = float(np.sum(assign_cost[np.arange(n_cust), res]))
        return open_sum + assign_sum
    
    def is_valid(res, load):
        """Check if solution is valid (no capacity violations, indices in range)."""
        if res is None:
            return False
        if np.any(res < 0) or np.any(res >= n_fac):
            return False
        if np.any(load > capacities):
            return False
        if np.any(load < 0):
            return False
        # Ensure every customer is assigned
        if len(res) != n_cust:
            return False
        return True
    
    def loads_of(res):
        load = np.zeros(n_fac, dtype=np.int64)
        np.add.at(load, res, demands)
        return load

    def local_search(res, load, max_time=5.0):
        """
        Refine solution with local search moves: single moves, swaps, and close-facility.
        Returns refined (res, load, score).
        """
        res = res.copy()
        load = load.copy()
        start_time = time.time()
        best_score = calc_cost(res)
        
        improved = True
        while improved and (time.time() - start_time) < max_time and time.time() < deadline - 2.0:
            improved = False
            
            # --- 1. Single Move Pass (Best Improvement) ---
            # We iterate through customers and find the best move for each.
            # To speed up, we only consider moves that might be beneficial.
            any_improve = False
            for i in range(n_cust):
                if time.time() > deadline - 2.0:
                    break
                f_old = res[i]
                d_i = demands[i]
                
                # Find feasible target facilities
                # A facility f is feasible if load[f] + d_i <= capacities[f]
                feasible_mask = (load + d_i <= capacities)
                feasible_mask[f_old] = False # Exclude current
                feasible = np.where(feasible_mask)[0]
                
                if len(feasible) == 0:
                    continue
                
                # Compute delta for each feasible target
                # Delta = (new_assign_cost - old_assign_cost) + open_cost_change
                base_assign = assign_cost[i, f_old]
                deltas = assign_cost[i, feasible] - base_assign
                
                # Open cost adjustment
                # If f_old becomes empty, we save open_cost[f_old]
                if load[f_old] == d_i:
                    deltas = deltas - open_cost[f_old]
                
                # If a new facility is opened, we pay open_cost[f_new]
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
                # If no single move improved, try swaps and closes
                
                # --- 2. Swap Moves (Sampled) ---
                # Randomly sample pairs of customers and check if swapping facilities improves cost.
                rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
                n_swap_samples = 2000
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
                    # f_i receives d_j, loses d_i. f_j receives d_i, loses d_j.
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    
                    # Compute delta
                    delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                   (assign_cost[j, f_i] - assign_cost[j, f_j])
                    
                    # Open cost: check if any facility becomes empty or opens
                    old_load_i = load[f_i]
                    old_load_j = load[f_j]
                    new_load_i = old_load_i - d_i + d_j
                    new_load_j = old_load_j - d_j + d_i
                    
                    # Open cost change
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
                
                # --- 3. Close Facility Move ---
                # Try to close each facility that is open and see if we can reassign its customers
                # to other facilities without increasing cost too much.
                for f_close in range(n_fac):
                    if time.time() > deadline - 2.0:
                        break
                    if load[f_close] == 0:
                        continue
                    
                    custs = np.where(res == f_close)[0]
                    if len(custs) == 0:
                        continue
                    
                    # Try to reassign all customers of f_close to other facilities
                    # We use a greedy approach: sort customers by demand descending or by cost
                    # and assign them to the cheapest feasible facility (excluding f_close)
                    
                    temp_load = load.copy()
                    temp_res = res.copy()
                    temp_load[f_close] = 0
                    
                    success = True
                    # Sort customers by demand descending to handle large demands first
                    order = custs[np.argsort(-demands[custs])]
                    
                    for c in order:
                        d_c = demands[c]
                        # Feasible facilities are those with remaining capacity, excluding f_close
                        feasible_mask = (temp_load + d_c <= capacities)
                        feasible_mask[f_close] = False
                        feasible = np.where(feasible_mask)[0]
                        
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
                        
                        # Open cost change
                        # f_close was open, now closed. Other facilities might open.
                        old_open = float(np.sum(open_cost[load > 0]))
                        new_open = float(np.sum(open_cost[temp_load > 0]))
                        
                        delta = (new_assign_sum - old_assign_sum) + (new_open - old_open)
                        
                        if delta < -1e-9:
                            res[:] = temp_res
                            load[:] = temp_load
                            improved = True
        
        final_score = calc_cost(res)
        return res, load, final_score

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
            # Greedy with open cost penalty (Lagrangian-like)
            # Penalize opening new facilities to encourage consolidation
            # Use a dynamic penalty based on average open cost
            avg_open = np.mean(open_cost)
            eff_cost = assign_cost.copy()
            order = np.argsort(-demands)
            for i in order:
                d_i = demands[i]
                feasible = np.where(load + d_i <= capacities)[0]
                if len(feasible) == 0:
                    return None
                costs = eff_cost[i, feasible].copy()
                # Add penalty for opening new facilities
                for idx, f_new in enumerate(feasible):
                    if load[f_new] == 0:
                        costs[idx] += avg_open
                f = int(feasible[np.argmin(costs)])
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
            # Min-cost flow inspired: Iterative refinement
            # Start with a random assignment, then refine by solving a smaller problem
            # This is a simplified version. We will use a heuristic that balances load.
            # Assign to facility with lowest (assign_cost + load/capacity * penalty)
            order = np.argsort(-demands)
            penalty = np.mean(open_cost) * 0.5
            for i in order:
                d_i = demands[i]
                feasible = np.where(load + d_i <= capacities)[0]
                if len(feasible) == 0:
                    return None
                # Score: assign cost + penalty for using capacity
                # If facility is empty, add full open cost
                scores = assign_cost[i, feasible].copy()
                for idx, f_new in enumerate(feasible):
                    if load[f_new] == 0:
                        scores[idx] += open_cost[f_new]
                    else:
                        # Small penalty for filling up capacity
                        scores[idx] += (load[f_new] / capacities[f_new]) * penalty
                f = int(feasible[np.argmin(scores)])
                res[i] = f
                load[f] += d_i
                
        elif method == 4:
            # Randomized greedy with multiple alpha values
            alpha = rng.uniform(0, np.mean(open_cost))
            eff_cost = assign_cost + alpha * np.outer(np.ones(n_cust), np.ones(n_fac) / n_fac) # Simple spread
            # Better: penalize opening
            order = np.argsort(-demands)
            for i in order:
                d_i = demands[i]
                feasible = np.where(load + d_i <= capacities)[0]
                if len(feasible) == 0:
                    return None
                costs = assign_cost[i, feasible].copy()
                for idx, f_new in enumerate(feasible):
                    if load[f_new] == 0:
                        costs[idx] += alpha
                f = int(feasible[np.argmin(costs)])
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
    n_candidates = 100
    methods = [0, 1, 2, 3, 4]
    
    for k in range(n_candidates):
        if time.time() > deadline - 15.0:
            break
        
        method = methods[k % len(methods)]
        candidate = generate_candidate(method, rng)
        
        if candidate is None:
            continue
        
        res, load = candidate
        
        # Compute initial score
        score = calc_cost(res)
        
        # Refine with local search: allocate time based on remaining budget
        remaining = deadline - time.time()
        if k < 10:
            ls_time = min(5.0, remaining * 0.05)
        elif k < 30:
            ls_time = min(3.0, remaining * 0.03)
        else:
            ls_time = min(1.5, remaining * 0.02)
            
        if ls_time < 0.2:
            ls_time = 0.2
            
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
                # If infeasible, pick facility with most remaining capacity
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
