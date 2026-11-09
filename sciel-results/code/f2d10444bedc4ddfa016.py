"""Rewritten program for ACO multiple-constraint knapsack heuristic design."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Return one attractiveness score per item for the ACO solver.
    
    prizes: shape (5, 100), uniform in [0,1]
    weights: shape (5, 100), uniform in [0,1]
    
    Strategy: We want items that give high prize while consuming little of each
    constraint. The ACO transition probability is proportional to pheromone[item] * heuristic[item].
    Since pheromone starts at 1 and is updated based on best solutions, the heuristic
    guides the initial search. We want to emphasize:
      - High total prize
      - Low total weight (sum across constraints)
      - Low max weight in any single constraint (to avoid being blocked by one tight constraint)
      - Good prize-to-weight ratio per constraint
    
    We combine several features:
      1. prize_sum / (weight_sum + eps)  -- overall efficiency
      2. prize_sum / (max_weight + eps)  -- robustness to the tightest constraint
      3. A multiplicative penalty for items that are heavy in any constraint
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Validate shape
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Expected shape (5, 100) for both prizes and weights")
    
    n_items = prizes.shape[1]
    
    prize_sum = prizes.sum(axis=0)       # shape (100,)
    weight_sum = weights.sum(axis=0)     # shape (100,)
    weight_max = weights.max(axis=0)     # shape (100,)
    
    eps = 1e-9
    
    # Feature 1: overall prize-to-weight ratio
    feat1 = prize_sum / (weight_sum + eps)
    
    # Feature 2: prize-to-max-weight ratio (robustness)
    feat2 = prize_sum / (weight_max + eps)
    
    # Feature 3: per-constraint efficiency, take the minimum (weakest link)
    # For each constraint c, efficiency_c = prize_c / (weight_c + eps)
    # The item's bottleneck is the minimum efficiency across constraints
    per_constraint_eff = prizes / (weights + eps)  # shape (5, 100)
    feat3 = per_constraint_eff.min(axis=0)  # shape (100,)
    
    # Feature 4: prize density with penalty for heavy items
    # Items with weight_sum > capacity/2 (i.e., > 12.5... but capacity is 25 and weights are in [0,1])
    # Actually weights are uniform in [0,1], so weight_sum is in [0,5]. Capacity is 25.
    # So most items fit easily. The binding constraint is more about total capacity.
    # Let's use a soft penalty: 1 / (1 + weight_sum^2) to slightly prefer lighter items
    feat4 = 1.0 / (1.0 + weight_sum * weight_sum)
    
    # Combine features with geometric mean (multiplicative) to balance
    # Normalize each feature to [0, 1] range first
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)
    
    f1 = normalize(feat1)
    f2 = normalize(feat2)
    f3 = normalize(feat3)
    f4 = normalize(feat4)
    
    # Geometric mean with weights
    # We want a strong signal. Use a power-weighted geometric mean.
    # log-space combination
    log_h = (np.log(f1 + eps) + np.log(f2 + eps) + np.log(f3 + eps) + 0.5 * np.log(f4 + eps)) / 3.5
    
    heuristic = np.exp(log_h)
    
    # Ensure all positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Scale to a reasonable range (larger is more attractive)
    # The ACO uses pheromone * heuristic, pheromone starts at 1.
    # Let's scale so that the mean is around 1, with a wide dynamic range
    heuristic = heuristic / (heuristic.mean() + eps)
    
    # Add a small floor to avoid zero
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
