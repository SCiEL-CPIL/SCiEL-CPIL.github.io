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
    """Largest demand first, cheapest feasible facility with optional noise."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    
    order = np.argsort(-demands)
    for i in order:
        # Add small random noise to break ties and diversify
        noise = rng.uniform(0, 1, n_fac) * 0.1
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
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
            else:
                # Should not happen if capacities are sufficient, but handle gracefully
                # Force assign to least loaded facility if possible, else invalid
                f = np.argmin(load)
                result[i] = f
                load[f] += demands[i]
    return result

def _best_improvement_local_search(demands, capacities, open_cost, assign_cost, result, time_limit=10.0):
    """
    Local search using Best-Improvement strategy.
    Evaluates all possible moves and swaps, applies the one with the largest cost reduction.
    """
    start_time = time.time()
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    current_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_cost):
        return result, current_cost

    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        best_delta = 1e-9
        best_move = None # (type, i, j, new_f_i, new_f_j)
        
        # 1. Evaluate all single moves
        for i in range(n_cust):
            current_f = result[i]
            # Pre-compute delta parts that don't depend on f
            # delta = assign_cost[i, f] - assign_cost[i, current_f]
            # + open_cost[f] if load[f]==0 else 0
            # - open_cost[current_f] if load[current_f]==demands[i] else 0
            
            current_open_penalty = open_cost[current_f] if load[current_f] == demands[i] else 0.0
            base_assign_cost = assign_cost[i, current_f]
            
            for f in range(n_fac):
                if f == current_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                
                delta = assign_cost[i, f] - base_assign_cost
                if load[f] == 0:
                    delta += open_cost[f]
                delta -= current_open_penalty
                
                if delta < best_delta:
                    best_delta = delta
                    best_move = ('move', i, -1, f, -1)
        
        # 2. Evaluate all pairwise swaps
        for i in range(n_cust):
            fi = result[i]
            for j in range(i + 1, n_cust):
                fj = result[j]
                if fi == fj:
                    continue
                
                # Check capacity feasibility
                new_load_fi = load[fi] - demands[i] + demands[j]
                new_load_fj = load[fj] - demands[j] + demands[i]
                
                if new_load_fi > capacities[fi] or new_load_fj > capacities[fj]:
                    continue
                
                # Calculate delta
                delta = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                
                # Open cost changes
                # fi becomes empty if new_load_fi == 0
                # fj becomes empty if new_load_fj == 0
                # Note: fi and fj were non-empty before swap (since they had i and j)
                
                if new_load_fi == 0:
                    delta -= open_cost[fi]
                if new_load_fj == 0:
                    delta -= open_cost[fj]
                
                if delta < best_delta:
                    best_delta = delta
                    best_move = ('swap', i, j, fj, fi)
        
        # Apply best move if found
        if best_move is not None:
            improved = True
            move_type = best_move[0]
            if move_type == 'move':
                _, i, _, new_f, _ = best_move
                old_f = result[i]
                load[old_f] -= demands[i]
                load[new_f] += demands[i]
                result[i] = new_f
            else: # swap
                _, i, j, new_fi, new_fj = best_move
                # best_move stores (type, i, j, new_facility_for_i, new_facility_for_j)
                # In the loop above, we set best_move = ('swap', i, j, fj, fi)
                # So new_fi is fj, new_fj is fi
                fi = result[i]
                fj = result[j]
                
                load[fi] -= demands[i]
                load[fi] += demands[j]
                load[fj] -= demands[j]
                load[fj] += demands[i]
                
                result[i] = new_fi
                result[j] = new_fj
            
            current_cost += best_delta
            
            # Safety check: if cost becomes inf or negative (due to float error), break
            if current_cost < 0:
                current_cost = 0
            if np.isinf(current_cost):
                break

    return result, current_cost

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign customers to facilities using multi-start best-improvement local search.
    """
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    best_result = None
    best_cost = np.inf
    
    # Try multiple diverse initializations
    # We will run a few iterations of init + local search
    # Each local search has a time limit. We run as many as we can fit.
    
    num_starts = 0
    while time.time() < deadline:
        # Check if we have enough time for at least a quick local search
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        # If remaining time is very small, just return best found so far
        if remaining < 2.0:
            break
            
        # Determine time limit for local search
        # Allocate most of the remaining time to local search, keep a bit for next init
        # For the first few starts, give it more time.
        if num_starts == 0:
            ls_time_limit = max(1.0, remaining * 0.8)
        else:
            # For subsequent starts, we can be more aggressive if we are already good
            # But we need to ensure we don't waste time on bad starts.
            ls_time_limit = max(1.0, min(remaining * 0.7, 20.0)) # Cap at 20s per LS to allow multiple starts
        
        seed = num_starts * 100 + 42
        init_result = _greedy_init(demands, capacities, open_cost, assign_cost, seed=seed)
        
        # Quick check: if init is already very good, maybe skip heavy LS?
        # No, LS is the main optimizer.
        
        ls_result, ls_cost = _best_improvement_local_search(
            demands, capacities, open_cost, assign_cost, init_result, time_limit=ls_time_limit
        )
        
        if ls_cost < best_cost:
            best_cost = ls_cost
            best_result = ls_result.copy()
            
        num_starts += 1
        
        # If we found a perfect or near-perfect solution, we can stop early?
        # Hard to know "perfect", but if cost stops improving significantly, we might stop.
        # For now, just keep going until time runs out.
        
        # Optimization: If we have done several starts and the best cost isn't improving, 
        # we might want to increase the time limit for the next LS or stop.
        # Let's just let the time limit handle it.

    if best_result is None:
        # Fallback: simple greedy if something went wrong
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost, seed=0)
        
    return best_result

# EVOLVE-BLOCK-END
