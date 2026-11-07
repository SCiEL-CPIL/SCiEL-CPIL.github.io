"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ant colony optimization of multiple-constraint knapsack.
    
    Parameters:
    - prizes: (5, 100) array of prizes for 5 constraints, 100 items
    - weights: (5, 100) array of weights for 5 constraints, 100 items
    
    Returns:
    - array of 100 scores (larger is more attractive)
    """
    start_time = time.time()
    deadline = start_time + 290  # Leave safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = prizes.shape
    
    # Basic validation
    if prizes.shape != weights.shape:
        raise ValueError("prizes and weights must have the same shape")
    
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)  # (100,)
    
    # Total weight per item across all constraints
    total_weight = weights.sum(axis=0)  # (100,)
    
    # Per-constraint weight analysis
    # For each constraint, compute the relative scarcity
    # Items that consume more of a scarce constraint should be penalized
    
    # Capacity is 25 for each constraint
    capacity = 25.0
    
    # For each constraint c, the total weight of all items
    total_weight_per_constraint = weights.sum(axis=1)  # (5,)
    
    # Normalized weight per constraint: how much of the capacity this item uses
    # If an item uses a large fraction of a tight constraint, it's less attractive
    
    # Compute the "tightness" of each constraint: total_weight / capacity
    tightness = total_weight_per_constraint / capacity  # (5,)
    
    # For each item, compute a weighted penalty based on how much it consumes
    # of each constraint relative to its tightness
    # An item that consumes a lot of a very tight constraint is penalized more
    
    # normalized_weight[c, i] = weights[c, i] / capacity
    normalized_weight = weights / capacity  # (5, 100)
    
    # Weight the normalized weights by tightness
    # This emphasizes constraints that are already heavily loaded
    weighted_normalized = normalized_weight * tightness[:, np.newaxis]  # (5, 100)
    
    # Sum across constraints to get a total "cost" score
    total_cost = weighted_normalized.sum(axis=0)  # (100,)
    
    # Heuristic: prize density adjusted for constraint pressure
    # Higher total prize and lower total cost = more attractive
    
    # Use a smooth combination: prize / (1 + cost)
    # This ensures positive values and smooth behavior
    heuristic = total_prize / (1.0 + total_cost)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    # Try to refine with a local search / greedy approach if time permits
    # Simulate a few greedy passes to estimate item attractiveness
    
    elapsed = time.time() - start_time
    if elapsed < 20:  # If we have time, do a quick greedy simulation
        # Greedy simulation: repeatedly pick the best item that still fits
        # Track how often each item is picked as a proxy for attractiveness
        
        n_sims = 50  # Number of greedy simulations
        
        # Track selection frequency
        selection_count = np.zeros(n_items, dtype=np.float64)
        
        for sim in range(n_sims):
            if time.time() > deadline:
                break
            
            # Current remaining capacity
            remaining_cap = np.full(n_constraints, capacity, dtype=np.float64)
            
            # Available items
            available = np.ones(n_items, dtype=bool)
            
            # Greedy selection
            while True:
                # Find items that still fit
                fits = np.all(weights[available] <= remaining_cap[:, np.newaxis], axis=0)
                if not fits.any():
                    break
                
                # Get indices of fitting items
                fitting_indices = np.where(available & fits)[0]
                
                # Score each fitting item: prize / weight (density)
                # Use the same heuristic as a tiebreaker
                item_scores = np.zeros(n_items, dtype=np.float64)
                for i in fitting_indices:
                    # Prize density: total prize / total weight
                    if total_weight[i] > 1e-10:
                        item_scores[i] = total_prize[i] / total_weight[i]
                    else:
                        item_scores[i] = total_prize[i]
                
                # Pick the item with highest score
                best_idx = fitting_indices[np.argmax(item_scores[fitting_indices])]
                
                # Select it
                selection_count[best_idx] += 1
                remaining_cap -= weights[:, best_idx]
                available[best_idx] = False
                
                # Early stop if no more items fit
                if not np.all(weights[available] <= remaining_cap[:, np.newaxis], axis=0).any():
                    break
        
        # Normalize selection counts
        if selection_count.sum() > 0:
            selection_freq = selection_count / selection_count.sum()
            # Combine with original heuristic: geometric mean
            combined = np.sqrt(heuristic * (selection_freq + 1e-10))
            combined = np.where(np.isfinite(combined) & (combined > 0), combined, 1e-9)
            heuristic = combined
    
    return heuristic
# EVOLVE-BLOCK-END
