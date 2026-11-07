"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    timeout = 145.0  # Safety margin below 150s
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Convert to numpy arrays for ease of use
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    # Helper to check time
    def time_exceeded():
        return time.time() - start_time > timeout

    # Helper to calculate total cost
    def calc_total_cost(result, load):
        open_cost_sum = np.sum(open_cost[load > 0])
        assign_cost_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_cost_sum + assign_cost_sum

    def greedy_init_largest_demand():
        """Greedy initialization: Largest Demand First"""
        order = np.argsort(-demands)
        res = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=float)
        
        for i in order:
            d = demands[i]
            remaining_cap = capacities - load
            feasible_mask = remaining_cap >= d
            if not np.any(feasible_mask):
                # If no facility has enough capacity, pick the one with most remaining
                # This might lead to infeasibility if total demand > total capacity, 
                # but problem statement implies feasible instances.
                f = np.argmax(remaining_cap)
            else:
                costs = np.where(feasible_mask, assign_cost[i], np.inf)
                f = np.argmin(costs)
            
            res[i] = f
            load[f] += d
        
        return res, load

    def greedy_init_nearest():
        """Greedy initialization: Nearest Feasible Facility"""
        res = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=float)
        
        # Sort customers by their minimum possible assignment cost (distance to nearest facility)
        min_costs = np.min(assign_cost, axis=1)
        order = np.argsort(min_costs)
        
        for i in order:
            d = demands[i]
            remaining_cap = capacities - load
            feasible_mask = remaining_cap >= d
            if not np.any(feasible_mask):
                f = np.argmax(remaining_cap)
            else:
                costs = np.where(feasible_mask, assign_cost[i], np.inf)
                f = np.argmin(costs)
            
            res[i] = f
            load[f] += d
        
        return res, load

    def local_search(result, load):
        """
        Local search with 1-move (Relocation) and 2-move (Swap) improvements.
        Uses Best-Improvement strategy: finds the single best move in a full scan, applies it, repeats.
        """
        current_cost = calc_total_cost(result, load)
        best_cost = current_cost
        best_result = result.copy()
        best_load = load.copy()
        
        improved = True
        max_iterations = 1000 # Safety cap for infinite loops if logic errors
        iter_count = 0
        
        while improved and not time_exceeded() and iter_count < max_iterations:
            iter_count += 1
            improved = False
            best_delta = 0.0
            best_move_type = None
            best_move_data = None
            
            # --- Phase A: Check 1-moves (Relocation) ---
            # Vectorized approach for speed? 
            # For each i, we want to find j that minimizes delta.
            # delta = assign_cost[i, j] - assign_cost[i, f_old] + delta_open
            
            for i in range(n_customers):
                if time_exceeded():
                    return best_result, best_load, best_cost
                
                f_old = result[i]
                d = demands[i]
                
                # Pre-compute costs for all potential new facilities
                # We need to check capacity: load[j] + d <= capacities[j]
                feasible_mask = (capacities - load) >= d
                feasible_mask[f_old] = False # Can't move to same facility
                
                if not np.any(feasible_mask):
                    continue
                    
                # Calculate delta for all feasible j
                # Base delta from assignment cost
                base_delta = assign_cost[i, feasible_mask] - assign_cost[i, f_old]
                
                # Calculate open cost delta
                # If f_old becomes empty, we save open_cost[f_old]
                # If f_new was empty, we pay open_cost[f_new]
                
                # This logic is tricky to fully vectorize without care for the "empty" condition.
                # Let's stick to loop for correctness and simplicity, it's fast enough for N=100.
                
                for f_new in np.where(feasible_mask)[0]:
                    if load[f_new] + d > capacities[f_new]:
                        continue
                        
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    delta_open = 0.0
                    
                    if load[f_old] == d:
                        delta_open -= open_cost[f_old]
                    
                    if load[f_new] == 0:
                        delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_move_type = 1
                        best_move_data = (i, f_old, f_new)
            
            # --- Phase B: Check 2-moves (Swap) ---
            for i in range(n_customers):
                if time_exceeded():
                    return best_result, best_load, best_cost
                
                f_i = result[i]
                d_i = demands[i]
                
                for j in range(i + 1, n_customers):
                    if time_exceeded():
                        return best_result, best_load, best_cost
                    
                    f_j = result[j]
                    d_j = demands[j]
                    
                    if f_i == f_j:
                        continue
                    
                    new_load_i = load[f_i] - d_i + d_j
                    new_load_j = load[f_j] - d_j + d_i
                    
                    if new_load_i > capacities[f_i] or new_load_j > capacities[f_j]:
                        continue
                    
                    delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - \
                                   (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    old_open = (open_cost[f_i] if load[f_i] > 0 else 0.0) + \
                               (open_cost[f_j] if load[f_j] > 0 else 0.0)
                    
                    new_open = (open_cost[f_i] if new_load_i > 0 else 0.0) + \
                               (open_cost[f_j] if new_load_j > 0 else 0.0)
                    
                    delta_open = new_open - old_open
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_move_type = 2
                        best_move_data = (i, j, f_i, f_j)
            
            # Apply best move if found
            if best_move_type is not None:
                if best_move_type == 1:
                    i, f_old, f_new = best_move_data
                    load[f_old] -= demands[i]
                    load[f_new] += demands[i]
                    result[i] = f_new
                else:
                    i, j, f_i, f_j = best_move_data
                    d_i = demands[i]
                    d_j = demands[j]
                    load[f_i] -= d_i
                    load[f_j] -= d_j
                    load[f_i] += d_j
                    load[f_j] += d_i
                    result[i] = f_j
                    result[j] = f_i
                
                current_cost += best_delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = result.copy()
                    best_load = load.copy()
                improved = True
                
        return best_result, best_load, best_cost

    def perturb(result, load, rng, strength=10):
        """
        Perturb the solution by performing 'strength' random valid moves.
        Returns new result and load.
        """
        res = result.copy()
        load = load.copy()
        
        for _ in range(strength):
            if time_exceeded():
                break
            
            move_type = rng.integers(0, 2) # 0: relocate, 1: swap
            
            if move_type == 0:
                i = rng.integers(0, n_customers)
                f_old = res[i]
                d = demands[i]
                
                feasible = np.where((capacities - load) >= d)[0]
                feasible = feasible[feasible != f_old]
                
                if len(feasible) > 0:
                    f_new = feasible[rng.integers(0, len(feasible))]
                    load[f_old] -= d
                    load[f_new] += d
                    res[i] = f_new
            else:
                i = rng.integers(0, n_customers)
                j = rng.integers(0, n_customers)
                
                if i == j or res[i] == res[j]:
                    continue
                
                f_i = res[i]
                f_j = res[j]
                d_i = demands[i]
                d_j = demands[j]
                
                new_load_i = load[f_i] - d_i + d_j
                new_load_j = load[f_j] - d_j + d_i
                
                if new_load_i > capacities[f_i] or new_load_j > capacities[f_j]:
                    continue
                
                load[f_i] -= d_i
                load[f_j] -= d_j
                load[f_i] += d_j
                load[f_j] += d_i
                res[i] = f_j
                res[j] = f_i
                
        return res, load

    # --- Main Loop ---
    
    # 1. Try multiple initializations
    best_result = None
    best_load = None
    best_cost = float('inf')
    
    # Init 1: Largest Demand First
    res, load = greedy_init_largest_demand()
    res_ls, load_ls, cost_ls = local_search(res, load)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_result = res_ls
        best_load = load_ls
        
    # Init 2: Nearest Facility First
    if not time_exceeded():
        res, load = greedy_init_nearest()
        res_ls, load_ls, cost_ls = local_search(res, load)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_result = res_ls
            best_load = load_ls

    # 2. Random Restarts with Perturbation
    rng = np.random.default_rng(42)
    restart_count = 0
    max_restarts = 100 # Increased restarts
    
    while not time_exceeded() and restart_count < max_restarts:
        # Perturb current best solution
        # Increase perturbation strength slightly as time goes on? Or keep constant.
        # Let's try varying strength: start small, then larger.
        strength = 5 + int(10 * (restart_count / max_restarts))
        
        res_pert, load_pert = perturb(best_result, best_load, rng, strength)
        
        # Run local search from perturbed solution
        res_ls, load_ls, cost_ls = local_search(res_pert, load_pert)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_result = res_ls
            best_load = load_ls
            
        restart_count += 1
        
        # If we've exhausted good restarts (no improvement in last 20), break early?
        # Or just let it run until time limit.
        
    return best_result.astype(int)
# EVOLVE-BLOCK-END
