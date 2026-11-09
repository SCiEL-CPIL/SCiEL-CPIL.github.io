"""Capacitated facility location: Multi-start local search with effective cost heuristics."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using a multi-start local search strategy.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Safety margin: 150s limit, aim for 145s max to be safe
    deadline = time.time() + 145.0
    
    def calc_score(res):
        """Calculate total cost for a given assignment."""
        if res is None:
            return np.inf
        # Check validity
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[res[i]] += demands[i]
        if np.any(load > capacities):
            return np.inf
        used = np.unique(res)
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])

    def repair_capacity(res, load):
        """
        Repair capacity violations by moving customers from overloaded facilities
        to underloaded ones with minimum cost increase.
        Returns (res, load, valid).
        """
        res = res.copy()
        load = load.copy()
        
        max_iter = 500
        for _ in range(max_iter):
            if time.time() > deadline - 2.0:
                return res, load, np.all(load <= capacities)
            
            overloaded = np.where(load > capacities)[0]
            if len(overloaded) == 0:
                return res, load, True
                
            f = overloaded[0]
            # Find customers in f
            custs_in_f = np.where(res == f)[0]
            if len(custs_in_f) == 0:
                # Should not happen
                break
                
            best_cust = -1
            best_g = -1
            min_penalty = np.inf
            
            for i in custs_in_f:
                for g in range(n_fac):
                    if g == f:
                        continue
                    if load[g] + demands[i] <= capacities[g]:
                        penalty = assign_cost[i, g] - assign_cost[i, f]
                        if penalty < min_penalty:
                            min_penalty = penalty
                            best_cust = i
                            best_g = g
            
            if best_cust == -1:
                # No feasible move found
                return res, load, False
                
            # Perform move
            res[best_cust] = best_g
            load[f] -= demands[best_cust]
            load[best_g] += demands[best_cust]
            
        return res, load, np.all(load <= capacities)

    def local_search(res, load, max_time=10.0):
        """
        Perform local search: try moving each customer to the best alternative facility.
        Iterate until no improvement or time limit.
        """
        res = res.copy()
        load = load.copy()
        
        start_time = time.time()
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > max_time:
                break
            if time.time() > deadline - 1.0:
                break
                
            for i in range(n_cust):
                if time.time() - start_time > max_time:
                    break
                if time.time() > deadline - 1.0:
                    break
                    
                f_old = res[i]
                best_delta = np.inf
                best_f_new = -1
                
                for f_new in range(n_fac):
                    if f_new == f_old:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new]:
                        continue
                        
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    
                    delta_open = 0.0
                    if load[f_old] == demands[i]:
                        delta_open -= open_cost[f_old]
                    if load[f_new] == 0:
                        delta_open += open_cost[f_new]
                        
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new
                
                if best_f_new != -1 and best_delta < -1e-9:
                    res[i] = best_f_new
                    load[f_old] -= demands[i]
                    load[best_f_new] += demands[i]
                    improved = True
                    
        return res, load

    def swap_local_search(res, load, max_time=10.0):
        """
        Perform swap local search: try swapping assignments of two customers.
        """
        res = res.copy()
        load = load.copy()
        
        start_time = time.time()
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > max_time:
                break
            if time.time() > deadline - 1.0:
                break
                
            for i in range(n_cust):
                if time.time() - start_time > max_time:
                    break
                if time.time() > deadline - 1.0:
                    break
                    
                f_i = res[i]
                for j in range(i + 1, n_cust):
                    if time.time() - start_time > max_time:
                        break
                    if time.time() > deadline - 1.0:
                        break
                        
                    f_j = res[j]
                    if f_i == f_j:
                        continue
                        
                    # Check capacity after swap
                    if load[f_i] - demands[i] + demands[j] > capacities[f_i]:
                        continue
                    if load[f_j] - demands[j] + demands[i] > capacities[f_j]:
                        continue
                        
                    # Calculate cost delta
                    delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                   (assign_cost[j, f_i] - assign_cost[j, f_j])
                    
                    delta_open = 0.0
                    # f_i: loses i, gains j
                    # f_j: loses j, gains i
                    # Open cost only changes if a facility becomes empty or was empty
                    # But since both facilities already have customers (f_i has i, f_j has j),
                    # and after swap they still have customers (f_i has j, f_j has i),
                    # no facility becomes empty or newly opened.
                    # So delta_open = 0.
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        # Perform swap
                        res[i], res[j] = res[j], res[i]
                        load[f_i] -= demands[i]
                        load[f_i] += demands[j]
                        load[f_j] -= demands[j]
                        load[f_j] += demands[i]
                        improved = True
                        break  # Restart outer loop for this i? Or just continue?
                        # Let's just mark improved and continue checking.
                        
        return res, load

    # Initialize best solution with greedy
    best_score = np.inf
    best_result = np.zeros(n_cust, dtype=int)
    
    # Greedy initialization: largest demand first, cheapest feasible facility
    result_greedy = np.zeros(n_cust, dtype=int)
    load_greedy = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        feasible = np.where(load_greedy + demands[i] <= capacities)[0]
        if len(feasible) == 0:
            # Fallback: pick facility with max remaining capacity
            remaining = capacities - load_greedy
            f = np.argmax(remaining)
            result_greedy[i] = f
            load_greedy[f] += demands[i]
        else:
            costs = assign_cost[i, feasible]
            f_idx = feasible[np.argmin(costs)]
            result_greedy[i] = f_idx
            load_greedy[f_idx] += demands[i]
            
    best_result = result_greedy.copy()
    best_score = calc_score(best_result)
    
    # Multi-start with different alpha values for effective cost
    alphas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    for alpha in alphas:
        if time.time() > deadline - 15.0:
            break
            
        # Effective cost: assign_cost + alpha * open_cost
        eff_cost = assign_cost + alpha * open_cost
        init_res = np.argmin(eff_cost, axis=1)
        
        # Calculate initial load
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[init_res[i]] += demands[i]
            
        # Repair capacity violations
        res, load, valid = repair_capacity(init_res, load)
        
        if not valid:
            continue
            
        # Local search
        res, load = local_search(res, load, max_time=5.0)
        
        # Swap local search
        res, load = swap_local_search(res, load, max_time=3.0)
        
        # Calculate score
        score = calc_score(res)
        if score < best_score:
            best_score = score
            best_result = res.copy()
            
    # Final local search on best result
    load_best = np.zeros(n_fac)
    for i in range(n_cust):
        load_best[best_result[i]] += demands[i]
        
    best_result, load_best = local_search(best_result, load_best, max_time=20.0)
    best_result, load_best = swap_local_search(best_result, load_best, max_time=10.0)
    
    # Final check
    final_score = calc_score(best_result)
    if final_score < best_score:
        best_score = final_score
    
    return best_result
# EVOLVE-BLOCK-END
