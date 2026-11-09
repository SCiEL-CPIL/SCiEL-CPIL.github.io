import time
import numpy as np
import random

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns each customer to a facility using Lagrangian Relaxation for initialization
    and Tabu Search for refinement.
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
        This is a simple repair heuristic.
        """
        res = res.copy()
        load = load.copy()
        
        # Identify facilities with excess load
        excess_facs = np.where(load > capacities)[0]
        
        for f_excess in excess_facs:
            # Customers assigned to this facility
            custs = np.where(res == f_excess)[0]
            # Sort by demand descending to move big ones first? Or by cost?
            # Let's try to move customers to the facility with the most remaining capacity
            # that minimizes the cost increase.
            
            # We need to free up 'load[f_excess] - capacities[f_excess]' units of demand.
            # We will pick customers from f_excess and move them.
            
            # To make it fast, we just pick customers in reverse order of assignment or random
            # and move them to the best available facility.
            
            # Let's iterate through customers in f_excess
            # We need to remove enough demand.
            # Sort customers by demand descending to reduce the number of moves?
            # Or sort by the difference in cost to next best facility?
            
            # Simple strategy: move customers one by one until capacity is met.
            # Pick the customer with the smallest "penalty" (cost increase) to move.
            
            # Precompute for each customer in f_excess the best alternative
            # This might be expensive if done repeatedly.
            
            # Let's just iterate and move if it helps.
            # We'll use a priority queue logic or just sort.
            
            # For each customer c in f_excess:
            # Find best facility f' != f_excess with capacity.
            # Cost delta = assign_cost[c, f'] - assign_cost[c, f_excess] + (open_cost[f'] if load[f']==0 else 0) - (open_cost[f_excess] if load[f_excess]==demands[c] else 0)
            # This is complex. Let's just do a simple greedy repair.
            
            while load[f_excess] > capacities[f_excess]:
                # Find customer to move
                # Try to move the one with the smallest cost increase
                min_delta = np.inf
                best_c = -1
                best_f_new = -1
                
                for c in custs:
                    if res[c] != f_excess: continue # already moved
                    d_c = demands[c]
                    if d_c > capacities.max(): # Cannot fit anywhere? Problem instance should be feasible
                        continue
                    
                    # Find best target
                    feasible = np.where(load + d_c <= capacities)[0]
                    if len(feasible) == 0:
                        # No capacity anywhere? This shouldn't happen if total demand <= total capacity
                        # If it does, we are in trouble. Return current best?
                        break
                    
                    # Exclude f_excess from target? No, we are moving away from it.
                    # But we need to check if moving c to f' is valid.
                    # load[f_excess] - d_c <= capacities[f_excess] is always true if we are fixing excess.
                    
                    # Calculate delta
                    # Old cost part: assign_cost[c, f_excess]
                    # New cost part: assign_cost[c, f'] + open_cost[f'] if load[f']==0
                    # Note: open_cost[f_excess] might drop if load becomes 0, but we are moving away, so load decreases.
                    # If load[f_excess] becomes 0, we save open_cost[f_excess].
                    
                    # Simplified: just look at assignment cost for now, refine later.
                    # Actually, let's compute full delta.
                    
                    old_load_f = load[f_excess]
                    new_load_f = old_load_f - d_c
                    
                    for f_new in feasible:
                        if f_new == f_excess:
                            continue
                        new_load_f_new = load[f_new] + d_c
                        
                        # Cost change
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
                    # Could not find a move? Break to avoid infinite loop
                    break
                
                # Perform move
                res[best_c] = best_f_new
                load[f_excess] -= demands[best_c]
                load[best_f_new] += demands[best_c]
                
                # Update custs list? It's a numpy array, we can just check res[best_c] != f_excess in next iter
                # Or remove it.
                
        return res, load

    def lagrangian_init(max_time=5.0):
        """
        Use Lagrangian Relaxation to find a good initial solution.
        """
        start_time = time.time()
        best_res = None
        best_score = np.inf
        best_load = None
        
        # Initial multipliers
        lam = np.zeros(n_fac, dtype=np.float64)
        
        # Step size
        step_size = 1.0
        max_iter = 1000
        
        # Precompute costs
        # For each customer i, cost to facility f is assign_cost[i, f] + lam[f] * demands[i]
        
        for iter in range(max_iter):
            if time.time() > start_time + max_time:
                break
            
            # 1. Solve relaxed problem
            # For each customer i, choose f minimizing assign_cost[i, f] + lam[f] * demands[i]
            # This is a vectorized operation
            # costs_relaxed = assign_cost + np.outer(demands, lam)
            # res_relaxed = np.argmin(costs_relaxed, axis=1)
            
            # To save memory/time, we can do it in chunks or just use numpy broadcasting
            # 100x20 is small, so broadcasting is fine.
            costs_relaxed = assign_cost + np.outer(demands, lam)
            res_relaxed = np.argmin(costs_relaxed, axis=1)
            
            # 2. Calculate load
            load_relaxed = np.zeros(n_fac, dtype=np.int64)
            np.add.at(load_relaxed, res_relaxed, demands)
            
            # 3. Update multipliers (Subgradient)
            # Subgradient g_f = load_relaxed[f] - capacities[f]
            g = load_relaxed - capacities
            g = np.where(g > 0, g, 0) # Only positive violations contribute to subgradient for minimization?
            # Actually, the subgradient of the Lagrangian dual is (load - capacity).
            # If load > capacity, we need to increase lam to penalize this facility.
            # If load < capacity, we can decrease lam.
            # Standard subgradient: lam_new = max(0, lam + step * (load - capacity))
            
            # Normalize subgradient
            norm_g = np.linalg.norm(g)
            if norm_g < 1e-6:
                # Convergence
                break
                
            # Step size
            # Use a diminishing step size
            # step = alpha * (UB - LB) / ||g||^2
            # We don't have LB directly, but we can approximate.
            # For now, use a fixed step size that decays.
            step = 1.0 / (iter + 1)
            
            lam = np.maximum(0, lam + step * g)
            
            # 4. Check for feasible solution
            # The relaxed solution might be infeasible.
            # We can try to project it to a feasible solution.
            # If it's already feasible, it's a candidate.
            if np.all(load_relaxed <= capacities):
                score = calc_cost(res_relaxed)
                if score < best_score:
                    best_score = score
                    best_res = res_relaxed.copy()
                    best_load = load_relaxed.copy()
            else:
                # Try to repair
                res_feas, load_feas = make_feasible(res_relaxed, load_relaxed)
                if is_valid(res_feas, load_feas):
                    score = calc_cost(res_feas)
                    if score < best_score:
                        best_score = score
                        best_res = res_feas.copy()
                        best_load = load_feas.copy()
                        
        # If no feasible solution found, fall back to greedy
        if best_res is None:
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
            best_res = res
            best_load = load
            
        return best_res, best_load

    def tabu_search(res, load, max_time=10.0):
        """
        Tabu Search local search.
        """
        res = res.copy()
        load = load.copy()
        best_res = res.copy()
        best_load = load.copy()
        best_score = calc_cost(res)
        current_score = best_score
        
        start_time = time.time()
        tabu_tenure = 20
        tabu_list = {} # key: (i, f_old, f_new), value: time_expires
        # Or simpler: list of (i, f) that are tabu for customer i
        # tabu[i] = set of facilities tabu for customer i
        tabu = [set() for _ in range(n_cust)]
        
        iteration = 0
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() < start_time + max_time and time.time() < deadline - 5.0:
            iteration += 1
            
            # Find best move
            best_move = None
            best_move_score = current_score
            best_i = -1
            best_f_old = -1
            best_f_new = -1
            
            # Explore neighbors
            # 1. Single moves
            for i in range(n_cust):
                f_old = res[i]
                d_i = demands[i]
                
                # Check tabu
                # If (i, f_old, f_new) is in tabu, skip unless aspiration
                
                feasible = np.where(load + d_i <= capacities)[0]
                if len(feasible) == 0:
                    continue
                    
                # Exclude current facility
                mask = feasible != f_old
                feasible = feasible[mask]
                
                if len(feasible) == 0:
                    continue
                
                # Calculate deltas
                base_assign = assign_cost[i, f_old]
                deltas = assign_cost[i, feasible] - base_assign
                
                # Open cost changes
                if load[f_old] == d_i:
                    deltas = deltas - open_cost[f_old]
                    
                for idx, f_new in enumerate(feasible):
                    if load[f_new] == 0:
                        deltas[idx] += open_cost[f_new]
                    
                    # Check tabu
                    if f_new in tabu[i]:
                        # Aspiration: if this move improves best_score, accept
                        # We don't know the score yet, so we just mark it as tabu
                        # We can handle aspiration later or just skip for now
                        # Let's skip tabu moves for simplicity, aspiration is complex
                        continue
                        
                if len(deltas) == 0:
                    continue
                    
                best_idx = int(np.argmin(deltas))
                move_score = current_score + deltas[best_idx]
                
                if move_score < best_move_score:
                    best_move_score = move_score
                    best_i = i
                    best_f_old = res[i]
                    best_f_new = feasible[best_idx]
                    
            # 2. Swaps (optional, but adds power)
            # To save time, we might skip swaps in Tabu and rely on single moves + restarts
            # Or add a few random swaps
            
            if best_i != -1:
                # Perform move
                i = best_i
                f_old = best_f_old
                f_new = best_f_new
                d_i = demands[i]
                
                res[i] = f_new
                load[f_old] -= d_i
                load[f_new] += d_i
                current_score = best_move_score
                
                # Update tabu list
                # Add (i, f_old) to tabu? No, we tabu the move i -> f_new
                # So we tabu f_new for customer i
                tabu[i].add(f_new)
                
                # Remove oldest tabu entries if list is too long
                # Simple approach: clear tabu list every 'tabu_tenure' iterations
                if iteration % tabu_tenure == 0:
                    for i_t in range(n_cust):
                        tabu[i_t].clear()
                        
                # Check improvement
                if current_score < best_score:
                    best_score = current_score
                    best_res = res.copy()
                    best_load = load.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if no_improve_count > max_no_improve:
                    # Restart tabu list to escape local optimum
                    for i_t in range(n_cust):
                        tabu[i_t].clear()
                    no_improve_count = 0
            else:
                # No improving move found
                # Try a random move to escape
                if random.random() < 0.5:
                    i = random.randint(0, n_cust - 1)
                    f_old = res[i]
                    d_i = demands[i]
                    feasible = np.where(load + d_i <= capacities)[0]
                    feasible = feasible[feasible != f_old]
                    if len(feasible) > 0:
                        f_new = feasible[random.randint(0, len(feasible) - 1)]
                        res[i] = f_new
                        load[f_old] -= d_i
                        load[f_new] += d_i
                        current_score = calc_cost(res)
                        tabu[i].add(f_new)
                else:
                    break # Stuck, stop
                    
        # Final polish with greedy local search
        res, load = greedy_local_search(res, load, max_time=2.0)
        score = calc_cost(res)
        if score < best_score:
            best_score = score
            best_res = res.copy()
            best_load = load.copy()
            
        return best_res, best_load

    def greedy_local_search(res, load, max_time=5.0):
        """
        Standard greedy local search.
        """
        res = res.copy()
        load = load.copy()
        start_time = time.time()
        
        improved = True
        while improved and (time.time() - start_time) < max_time and time.time() < deadline - 3.0:
            improved = False
            
            # Single Move Pass
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
                # Swap Moves
                rng = np.random.default_rng(int(time.time() * 10000) % (2**32))
                n_swap_samples = 500
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
                        
        return res, load

    # 1. Initialization
    # Use Lagrangian Relaxation for a strong start
    res_init, load_init = lagrangian_init(max_time=5.0)
    
    # Also generate a few greedy starts for diversity
    rng = np.random.default_rng(123)
    
    def greedy_construct(order, open_penalty_scale=0.0):
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

    candidates = []
    candidates.append((res_init, load_init))
    
    # Greedy 1
    order1 = np.argsort(-demands)
    cand = greedy_construct(order1, 1.0)
    if cand: candidates.append(cand)
    
    # Greedy 2
    cand = greedy_construct(order1, 0.5)
    if cand: candidates.append(cand)
    
    # Greedy 3
    order3 = rng.permutation(n_cust)
    cand = greedy_construct(order3, 0.8)
    if cand: candidates.append(cand)
    
    # 2. Refine with Tabu Search
    best_result = None
    best_score = np.inf
    
    for idx, (res, load) in enumerate(candidates):
        if time.time() > deadline - 20.0:
            break
            
        # Allocate time for Tabu Search
        # First candidate gets more time
        if idx == 0:
            ts_time = min(20.0, deadline - time.time() - 10.0)
        else:
            ts_time = min(10.0, deadline - time.time() - 10.0)
            
        if ts_time < 1.0:
            ts_time = 1.0
            
        res_ts, load_ts = tabu_search(res, load, max_time=ts_time)
        score = calc_cost(res_ts)
        
        if is_valid(res_ts, load_ts) and score < best_score:
            best_score = score
            best_result = res_ts.copy()
            
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
            best_result, final_load = greedy_local_search(best_result, final_load, max_time=polish_time)
            
    # Final validation
    final_load = loads_of(best_result)
    if not is_valid(best_result, final_load):
        # Emergency repair
        best_result, final_load = make_feasible(best_result, final_load)
        
    return best_result.astype(int)
