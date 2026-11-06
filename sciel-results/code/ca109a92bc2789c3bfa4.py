"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _check_time():
    if _START_TIME is None:
        return True
    # Safety margin: 145s out of 150s limit
    return (time.time() - _START_TIME) < 145.0

def _fast_greedy_value(capacity, weights, values, indices, exclude_set):
    """
    Fast greedy fill using pre-sorted indices.
    Returns the total value obtained.
    """
    if capacity <= 0:
        return 0.0
    total_val = 0.0
    cap = capacity
    for idx in indices:
        if idx in exclude_set:
            continue
        w = weights[idx]
        if w <= cap:
            total_val += values[idx]
            cap -= w
            if cap <= 0:
                break
    return total_val

def _randomized_greedy_value(capacity, weights, values, indices, exclude_set, noise_level=0.1):
    """
    Simulates a greedy fill but with randomization in item selection.
    This helps explore different solution paths.
    """
    if capacity <= 0:
        return 0.0
    
    total_val = 0.0
    cap = capacity
    available_indices = [i for i in indices if i not in exclude_set]
    
    # Sort available indices by density, but add noise to allow different orders
    # We don't want to re-sort every time, so we pick from a subset of top items
    # Or simply: pick the best item from a random subset of the top K items?
    # A simpler approach: Sort by density once, then at each step, pick randomly 
    # from the top 'k' feasible items in the sorted list.
    
    # Pre-compute densities for available items? No, let's just use the pre-sorted 'indices'
    # and pick from the first few feasible ones with some probability.
    
    # Let's try a different randomization: 
    # At each step, consider the next feasible item in the sorted list. 
    # With probability p, swap it with a random other feasible item from the top 5.
    
    # For speed, let's just do a standard greedy but with a random tie-breaker 
    # or slight noise in the density comparison?
    # Actually, a common MCTS heuristic is to use a UCB-like formula, but here we are 
    # just estimating value. 
    # Let's stick to a simple randomized greedy: 
    # 1. Get feasible items from the sorted list.
    # 2. Pick one at random from the top 'k' feasible items (e.g., k=3 or 5).
    
    k_sample = 3
    while cap > 0:
        # Find the first k feasible items in the sorted order
        feasible_candidates = []
        for idx in indices:
            if idx in exclude_set:
                continue
            if weights[idx] <= cap:
                feasible_candidates.append(idx)
                if len(feasurable_candidates) >= k_sample:
                    break
        
        if not feasible_candidates:
            break
            
        # Pick a random one from the top k_sample
        chosen_idx = random.choice(feasible_candidates)
        total_val += values[chosen_idx]
        cap -= weights[chosen_idx]
        exclude_set.add(chosen_idx)
        
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    _init_time()
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Check time
    if not _check_time():
        # Fallback to simple greedy if time is tight
        densities = np.zeros(n)
        for i in range(n):
            if remaining_weights[i] > 0:
                densities[i] = remaining_values[i] / remaining_weights[i]
            else:
                densities[i] = np.inf
        feasible = np.where(remaining_weights <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        best_idx = -1
        best_dens = -1
        for i in feasible:
            if densities[i] > best_dens:
                best_dens = densities[i]
                best_idx = i
        return int(best_idx)

    # Identify feasible items
    feasible_indices = []
    for i in range(n):
        if remaining_weights[i] <= remaining_capacity:
            feasible_indices.append(i)
    
    if not feasible_indices:
        return -1

    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # Calculate densities
    densities = np.zeros(n)
    for i in range(n):
        if remaining_weights[i] > 0:
            densities[i] = remaining_values[i] / remaining_weights[i]
        else:
            densities[i] = np.inf
            
    # Sort indices by density descending
    sorted_indices = np.argsort(-densities)
    
    # Phase 1: 1-step lookahead to find top K candidates
    scores_1 = {}
    for i in feasible_indices:
        new_cap = remaining_capacity - remaining_weights[i]
        sim_val = remaining_values[i] + _fast_greedy_value(
            new_cap, 
            remaining_weights, 
            remaining_values, 
            sorted_indices, 
            {i}
        )
        scores_1[i] = sim_val
        
    # Determine K for MCTS phase
    # We want to spend most of the time on the MCTS simulations
    # Let's pick the top 10 candidates
    K = min(10, len(feasible_indices))
    sorted_candidates = sorted(scores_1.keys(), key=lambda x: scores_1[x], reverse=True)
    top_candidates = sorted_candidates[:K]
    
    # Phase 2: MCTS-like randomized greedy simulations
    # We will run multiple simulations for each top candidate
    # The number of simulations depends on the time budget and number of candidates
    
    # Estimate time per simulation
    # A simulation is O(n) roughly.
    # Let's set a target number of simulations per candidate
    # If we have 10 candidates, and 120s budget, we can do many simulations.
    # Let's try 100 simulations per candidate initially, and adjust if time is tight.
    
    sim_counts = {c: 0 for c in top_candidates}
    total_values = {c: 0.0 for c in top_candidates}
    
    # We will loop until time is up or we reach a certain number of sims
    # To be safe, let's do a fixed number of rounds of simulations
    # e.g., 50 rounds, in each round pick a random candidate and do 1 sim?
    # Or better: do N sims for each candidate.
    
    # Let's try 20 simulations per candidate.
    N_SIMS = 20
    
    for c in top_candidates:
        if not _check_time():
            break
            
        cap_after_c = remaining_capacity - remaining_weights[c]
        val_c = remaining_values[c]
        exclude_set = {c}
        
        for _ in range(N_SIMS):
            if not _check_time():
                break
                
            sim_val = val_c + _randomized_greedy_value(
                cap_after_c, 
                remaining_weights, 
                remaining_values, 
                sorted_indices, 
                exclude_set.copy()
            )
            total_values[c] += sim_val
            sim_counts[c] += 1
    
    # Calculate average value for each candidate
    best_item = top_candidates[0]
    best_avg_val = -1.0
    
    for c in top_candidates:
        if sim_counts[c] > 0:
            avg_val = total_values[c] / sim_counts[c]
            if avg_val > best_avg_val:
                best_avg_val = avg_val
                best_item = c
        else:
            # Fallback to 1-step score if no sims ran
            if scores_1[c] > best_avg_val:
                best_avg_val = scores_1[c]
                best_item = c

    return int(best_item)
# EVOLVE-BLOCK-END
