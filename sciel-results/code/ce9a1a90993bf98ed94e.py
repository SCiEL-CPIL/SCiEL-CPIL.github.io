"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost of an assignment."""
    n_cust = len(demands)
    n_fac = len(capacities)
    if len(result) != n_cust:
        return np.inf
    for i in range(n_cust):
        if result[i] < 0 or result[i] >= n_fac:
            return np.inf
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    for f in range(n_fac):
        if load[f] > capacities[f]:
            return np.inf
    total = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            total += open_cost[f]
    for i in range(n_cust):
        total += assign_cost[i, result[i]]
    return total


def _greedy_init(demands, capacities, open_cost, assign_cost, seed=None):
    """Largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    
    # Add small random perturbation to break ties and diversify
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    order = np.argsort(-demands)
    for i in order:
        # Add small random noise to assign_cost for this customer to break ties
        noise = rng.uniform(0, 1, n_fac) * 0.01
        adjusted_cost = assign_cost[i] + noise
        for f in np.argsort(adjusted_cost):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # Fallback: assign to facility with most remaining capacity
            rem = capacities - load
            f = np.argmax(rem)
            load[f] += demands[i]
            result[i] = f
    return result


def _local_search(demands, capacities, open_cost, assign_cost, result, time_limit=110.0, seed=None):
    """Local search: move customers to reduce cost while respecting capacity."""
    start_time = time.time()
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    best_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return result
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Randomize the order of customers to escape local optima
        cust_order = np.arange(n_cust)
        rng.shuffle(cust_order)
        
        for i in cust_order:
            if time.time() - start_time > time_limit:
                break
            current_f = result[i]
            
            # Calculate cost of moving customer i from current_f to each other facility
            for f in np.argsort(assign_cost[i]):
                if f == current_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                
                # Calculate delta cost
                delta = assign_cost[i, f] - assign_cost[i, current_f]
                
                # Check if current facility becomes empty
                if load[current_f] == demands[i]:
                    delta -= open_cost[current_f]
                
                # Check if new facility was empty
                if load[f] == 0:
                    delta += open_cost[f]
                
                if delta < -1e-9:
                    # Accept the move
                    load[current_f] -= demands[i]
                    load[f] += demands[i]
                    result[i] = f
                    best_cost += delta
                    improved = True
                    break  # Move this customer to the best feasible facility and break
    
    return result


def _swap_local_search(demands, capacities, open_cost, assign_cost, result, time_limit=110.0):
    """Local search: try swapping two customers between facilities."""
    start_time = time.time()
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    best_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return result
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Try swapping pairs of customers
        for i in range(n_cust):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_cust):
                if time.time() - start_time > time_limit:
                    break
                f_i = result[i]
                f_j = result[j]
                if f_i == f_j:
                    continue
                
                # Check if swap is feasible
                if load[f_i] - demands[i] + demands[j] > capacities[f_i]:
                    continue
                if load[f_j] - demands[j] + demands[i] > capacities[f_j]:
                    continue
                
                # Calculate delta cost
                delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                
                # Open cost changes: only if a facility becomes empty or was empty
                # f_i becomes empty if load[f_i] == demands[i]
                # f_j becomes empty if load[f_j] == demands[j]
                # f_i was empty before? No, it has customer i
                # f_j was empty before? No, it has customer j
                # After swap:
                # f_i has demand: load[f_i] - demands[i] + demands[j]
                # f_j has demand: load[f_j] - demands[j] + demands[i]
                
                # If f_i becomes empty after swap
                if load[f_i] - demands[i] + demands[j] == 0:
                    delta -= open_cost[f_i]
                # If f_j becomes empty after swap
                if load[f_j] - demands[j] + demands[i] == 0:
                    delta -= open_cost[f_j]
                # If f_i was not empty before but is now (always true since it had i)
                # If f_j was not empty before but is now (always true since it had j)
                # No new open costs since both facilities were already open
                
                if delta < -1e-9:
                    # Perform the swap
                    load[f_i] = load[f_i] - demands[i] + demands[j]
                    load[f_j] = load[f_j] - demands[j] + demands[i]
                    result[i] = f_j
                    result[j] = f_i
                    best_cost += delta
                    improved = True
                    break
            if improved:
                break
    
    return result


def assign(demands, capacities, open_cost, assign_cost):
    """Assign customers to facilities using multiple strategies and local search."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Try multiple initializations and keep the best
    best_result = None
    best_cost = np.inf
    
    # Strategy 1: Standard greedy
    result1 = _greedy_init(demands, capacities, open_cost, assign_cost, seed=42)
    cost1 = _compute_cost(result1, demands, capacities, open_cost, assign_cost)
    if cost1 < best_cost:
        best_cost = cost1
        best_result = result1.copy()
    
    # Strategy 2: Greedy with different random seed
    if time.time() < deadline - 10.0:
        result2 = _greedy_init(demands, capacities, open_cost, assign_cost, seed=123)
        cost2 = _compute_cost(result2, demands, capacities, open_cost, assign_cost)
        if cost2 < best_cost:
            best_cost = cost2
            best_result = result2.copy()
    
    # Strategy 3: Greedy with different random seed
    if time.time() < deadline - 10.0:
        result3 = _greedy_init(demands, capacities, open_cost, assign_cost, seed=789)
        cost3 = _compute_cost(result3, demands, capacities, open_cost, assign_cost)
        if cost3 < best_cost:
            best_cost = cost3
            best_result = result3.copy()
    
    # Local search on the best result
    if time.time() < deadline - 5.0:
        remaining_time = max(1.0, deadline - 5.0 - time.time())
        improved_result = _local_search(demands, capacities, open_cost, assign_cost, 
                                        best_result, time_limit=remaining_time * 0.7, seed=42)
        improved_cost = _compute_cost(improved_result, demands, capacities, open_cost, assign_cost)
        if improved_cost < best_cost:
            best_cost = improved_cost
            best_result = improved_result
    
    # Swap local search
    if time.time() < deadline - 5.0:
        remaining_time = max(1.0, deadline - 5.0 - time.time())
        swap_result = _swap_local_search(demands, capacities, open_cost, assign_cost, 
                                         best_result, time_limit=remaining_time)
        swap_cost = _compute_cost(swap_result, demands, capacities, open_cost, assign_cost)
        if swap_cost < best_cost:
            best_cost = swap_cost
            best_result = swap_result
    
    # Final local search pass
    if time.time() < deadline - 2.0:
        remaining_time = max(0.5, deadline - 2.0 - time.time())
        final_result = _local_search(demands, capacities, open_cost, assign_cost, 
                                     best_result, time_limit=remaining_time, seed=999)
        final_cost = _compute_cost(final_result, demands, capacities, open_cost, assign_cost)
        if final_cost < best_cost:
            best_cost = final_cost
            best_result = final_result
    
    return best_result
# EVOLVE-BLOCK-END
