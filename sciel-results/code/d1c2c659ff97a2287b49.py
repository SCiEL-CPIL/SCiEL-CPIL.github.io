"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores using a fast, deterministic, multi-start greedy simulation.
    
    Strategy:
    1. Perform multiple greedy passes with different item orderings (sorted by different criteria).
    2. In each pass, select items if they fit all constraints, tracking selection frequency.
    3. Weight selections by the total prize of the resulting solution.
    4. The final heuristic is the normalized weighted selection frequency.
    
    This captures complex constraint interactions and identifies items that are frequently 
    part of high-value feasible solutions, rather than relying on a simple static ratio.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    # Selection counts weighted by solution quality
    weighted_counts = np.zeros(n_items, dtype=float)
    
    # Time guard: ensure we don't exceed time limits
    start_time = time.time()
    max_time = 290.0  # Safety margin before 300s limit
    
    # Define several greedy strategies with different sorting criteria
    # Each strategy sorts items differently and greedily picks if they fit
    strategies = []
    
    # Strategy 1: Sort by prize/sum_weight (original heuristic idea)
    sum_w = weights.sum(axis=0)
    ratio1 = prizes / (sum_w + 1e-9)
    order1 = np.argsort(-ratio1)
    strategies.append(order1)
    
    # Strategy 2: Sort by prize/max_weight (binding constraint focus)
    max_w = weights.max(axis=0)
    ratio2 = prizes / (max_w + 1e-9)
    order2 = np.argsort(-ratio2)
    strategies.append(order2)
    
    # Strategy 3: Sort by prize only
    order3 = np.argsort(-prizes)
    strategies.append(order3)
    
    # Strategy 4: Sort by prize/(max_weight^2) - sharper
    ratio4 = prizes / (max_w ** 2 + 1e-9)
    order4 = np.argsort(-ratio4)
    strategies.append(order4)
    
    # Strategy 5: Sort by prize/min_weight (if min weight is low, item is "cheap")
    min_w = weights.min(axis=0)
    ratio5 = prizes / (min_w + 1e-9)
    order5 = np.argsort(-ratio5)
    strategies.append(order5)
    
    # Strategy 6: Sort by average prize per unit weight
    avg_w = weights.mean(axis=0)
    ratio6 = prizes / (avg_w + 1e-9)
    order6 = np.argsort(-ratio6)
    strategies.append(order6)
    
    # Strategy 7: Random-like ordering based on hash of index (deterministic)
    rng = np.random.RandomState(42)
    order7 = np.array(rng.permutation(n_items))
    strategies.append(order7)
    
    # Strategy 8: Another random ordering
    rng2 = np.random.RandomState(123)
    order8 = np.array(rng2.permutation(n_items))
    strategies.append(order8)
    
    # Run greedy simulations for each strategy
    for order in strategies:
        if time.time() - start_time > max_time:
            break
            
        # Greedy selection
        remaining_capacity = np.full(n_constraints, capacity, dtype=float)
        selected = []
        total_prize = 0.0
        
        for idx in order:
            # Check if item fits in all constraints
            if np.all(weights[:, idx] <= remaining_capacity):
                remaining_capacity -= weights[:, idx]
                selected.append(idx)
                total_prize += prizes[idx]
        
        # Weight the selections by the total prize of this solution
        # This rewards items that are part of high-value solutions
        if total_prize > 0:
            for idx in selected:
                weighted_counts[idx] += total_prize
    
    # Normalize and ensure positive values
    # Add a small base to avoid zeros, then normalize
    base = 1e-9
    scores = weighted_counts + base
    
    # Normalize to have similar scale as original heuristics
    if scores.max() > 0:
        scores = scores / scores.max()
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
