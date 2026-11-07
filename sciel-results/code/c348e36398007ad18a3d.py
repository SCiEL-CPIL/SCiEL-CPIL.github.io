import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    best_result = None
    best_score = float('inf')
    
    # Function to evaluate a solution
    def evaluate(result):
        load = np.zeros(n_facilities)
        for i, f in enumerate(result):
            load[f] += demands[i]
        if np.any(load > capacities + 1e-9):
            return float('inf')
        
        open_mask = np.zeros(n_facilities, dtype=bool)
        for f in result:
            open_mask[f] = True
        
        open_sum = np.sum(open_cost[open_mask])
        assign_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_sum + assign_sum

    # Greedy Initialization with optional randomization
    def greedy_init(randomize=False, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        is_open = np.zeros(n_facilities, dtype=bool)
        
        # Sort customers by demand descending
        order = np.argsort(-demands)
        
        for i in order:
            d = demands[i]
            # Feasible facilities
            feasible = load + d <= capacities
            
            if not np.any(feasible):
                # Should not happen if total capacity >= total demand, but handle gracefully
                # Fall back to cheapest
                f = np.argmin(assign_cost[i])
                result[i] = f
                load[f] += d
                is_open[f] = True
                continue
            
            # Calculate marginal costs
            # If facility is open, cost is assign_cost[i, f]
            # If closed, cost is assign_cost[i, f] + open_cost[f]
            marginal_costs = assign_cost[i].copy()
            closed_mask = ~is_open
            marginal_costs[closed_mask] += open_cost[closed_mask]
            
            # Mask infeasible
            marginal_costs[~feasible] = np.inf
            
            if randomize:
                # Pick from top 3
                sorted_indices = np.argsort(marginal_costs)
                top_k = sorted_indices[:min(3, n_facilities)]
                f = top_k[np.random.randint(len(top_k))]
            else:
                f = np.argmin(marginal_costs)
            
            result[i] = f
            load[f] += d
            is_open[f] = True
            
        return result, load, is_open

    # Local Search
    def local_search(result, load, is_open):
        # result: array of facility indices
        # load: array of current loads
        # is_open: array of boolean open status
        
        improved = True
        while improved:
            improved = False
            # Iterate customers in random order to break symmetry? Or fixed?
            # Fixed order is fine, maybe shuffle once per pass?
            for i in np.arange(n_customers):
                f_old = result[i]
                d = demands[i]
                
                best_delta = 1e-9
                best_f_new = -1
                
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    
                    # Check capacity
                    if load[f_new] + d > capacities[f_new]:
                        continue
                    
                    # Calculate delta
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    
                    delta_open = 0.0
                    # Check if old facility becomes empty
                    if load[f_old] - d == 0:
                        # It becomes empty, so we save its open cost
                        # But only if it was open (it must be, since it has customer i)
                        delta_open -= open_cost[f_old]
                        
                    # Check if new facility was empty
                    if load[f_new] == 0:
                        # It becomes non-empty, so we incur its open cost
                        # But only if it was closed
                        if not is_open[f_new]:
                            delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new
                        
                if best_f_new != -1:
                    # Apply move
                    # Update old facility
                    load[f_old] -= d
                    if load[f_old] == 0:
                        is_open[f_old] = False
                        
                    # Update new facility
                    load[best_f_new] += d
                    if load[best_f_new] == d: # Was 0
                        is_open[best_f_new] = True
                        
                    result[i] = best_f_new
                    improved = True
                    
                    # Time check inside loop? 
                    # 100 * 20 = 2000 ops per pass. Very fast.
                    if time.time() > deadline:
                        return result, load, is_open
                        
        return result, load, is_open

    # Main loop with restarts
    # 1. Deterministic greedy
    res, load, is_open = greedy_init(randomize=False)
    res, load, is_open = local_search(res, load, is_open)
    score = evaluate(res)
    if score < best_score:
        best_score = score
        best_result = res.copy()
        
    # 2. Randomized restarts
    # How many? 100 restarts * (Greedy ~1ms + LS ~5ms) = ~0.6s. Safe.
    # Let's do 50 restarts.
    for k in range(50):
        if time.time() > deadline - 10: # Leave 10s buffer
            break
            
        res, load, is_open = greedy_init(randomize=True, seed=k)
        res, load, is_open = local_search(res, load, is_open)
        score = evaluate(res)
        if score < best_score:
            best_score = score
            best_result = res.copy()
            
    return best_result
