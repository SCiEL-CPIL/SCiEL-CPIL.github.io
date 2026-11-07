"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item for the ant colony heuristic.
    
    Strategy: Constraint-regret rank product blended with harmonic-mean ratio
    and geometric slack, sharpened to emphasize items efficient across all constraints.
    
    The core idea is "constraint regret": for each constraint, an item's efficiency
    rank relative to all other items quantifies the opportunity cost of selecting it.
    The product of (1 - normalized_rank) across all constraints gives a score that
    is high only for items that are near-optimal in EVERY resource dimension.
    
    This is blended with:
      - Harmonic mean of prize/weight ratios (smooth min-ratio surrogate)
      - Geometric slack (rewards items consuming less capacity)
    
    The combined score is then sharpened via a power transform to emphasize
    high-efficiency items.
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    try:
        prizes = np.asarray(prizes, dtype=float)
        weights = np.asarray(weights, dtype=float)
        
        n_constraints, n_items = 5, 100
        capacity = 25.0
        eps = 1e-9
        
        # --- Component 1: Harmonic mean of per-constraint ratios ---
        safe_w = weights + eps
        ratios = prizes / safe_w  # (5, 100)
        inv_ratios = 1.0 / (ratios + eps)  # (5, 100)
        mean_inv = np.mean(inv_ratios, axis=0)  # (100,)
        h_harmonic = 1.0 / (mean_inv + eps)  # (100,)
        
        # --- Component 2: Geometric slack ---
        slack = 1.0 - weights / capacity  # (5, 100)
        slack = np.clip(slack, eps, 1.0)
        log_slack = np.log(slack)
        mean_log_slack = np.mean(log_slack, axis=0)  # (100,)
        h_slack = np.exp(mean_log_slack)  # (100,)
        
        # --- Component 3: Constraint-regret rank product ---
        # For each constraint c, rank items by efficiency (prize/weight).
        # Lower rank = better. Normalized rank in [0, 1].
        # Regret score per constraint = 1 - normalized_rank.
        # Product across constraints rewards items that are top-ranked in ALL constraints.
        
        # Efficiency per constraint: (5, 100)
        efficiency = prizes / safe_w  # same as ratios
        
        # Rank items within each constraint: 0 = best (highest efficiency)
        # argsort gives indices that would sort the array; we want rank
        # For each constraint, rank[i] = number of items with higher efficiency than item i
        ranks = np.zeros((n_constraints, n_items), dtype=float)
        for c in range(n_constraints):
            # Higher efficiency is better, so we want descending order
            # argsort of -efficiency gives indices in descending order of efficiency
            order = np.argsort(-efficiency[c])
            # rank of item i = position in the sorted order
            rank_array = np.zeros(n_items, dtype=float)
            rank_array[order] = np.arange(n_items, dtype=float)
            ranks[c] = rank_array
        
        # Normalize ranks to [0, 1]
        # rank 0 -> 0.0, rank n-1 -> 1.0
        normalized_ranks = ranks / (n_items - 1)  # (5, 100)
        
        # Regret score per constraint: 1 - normalized_rank
        regret_scores = 1.0 - normalized_ranks  # (5, 100), in [0, 1]
        
        # Product across constraints: high only if item is top-ranked in ALL
        # Use log-sum for numerical stability
        log_regret = np.log(regret_scores + eps)  # (5, 100)
        mean_log_regret = np.mean(log_regret, axis=0)  # (100,)
        h_regret = np.exp(mean_log_regret)  # (100,)
        
        # --- Blend the three components ---
        # Normalize each to [0, 1]
        def normalize_positive(arr):
            arr = np.where(arr > 0, arr, eps)
            mn = np.min(arr)
            mx = np.max(arr)
            if mx > mn:
                return (arr - mn) / (mx - mn)
            return np.ones_like(arr)
        
        h1 = normalize_positive(h_harmonic)  # (100,)
        h2 = normalize_positive(h_slack)     # (100,)
        h3 = normalize_positive(h_regret)    # (100,)
        
        # Blend: weight regret product heavily since it directly models opportunity cost
        alpha1 = 0.35  # harmonic mean
        alpha2 = 0.15  # geometric slack
        alpha3 = 0.50  # constraint-regret rank product
        
        h = alpha1 * h1 + alpha2 * h2 + alpha3 * h3
        
        # Ensure positive values
        h = np.where(h > 0, h, eps)
        
        # Sharpen: power transform to emphasize high-efficiency items
        # Power > 1 makes the distribution more peaked
        h = np.power(h + eps, 1.3)
        
        # Cleanup: replace non-finite or non-positive with eps
        h = np.where(np.isfinite(h) & (h > 0), h, eps)
        
        # Scale to reasonable range
        max_h = np.max(h)
        if max_h > 0:
            h = h / max_h
        
        # Ensure minimum value
        h = np.maximum(h, 1e-9)
        
        # Ensure shape is (100,)
        h = h.reshape(n_items)
        
        # Time check
        if time.time() > deadline:
            return h
            
        return h
        
    except Exception:
        # Fallback: return a valid array of small positive numbers
        return np.full((100,), 1e-9)
# EVOLVE-BLOCK-END
