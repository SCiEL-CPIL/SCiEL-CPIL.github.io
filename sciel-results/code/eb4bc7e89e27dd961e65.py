import numpy as np

def heuristics(prizes, weights):
    """
    Returns heuristic scores for items based on a Power Mean of per-constraint efficiencies.
    Uses p=-10 to strongly emphasize the bottleneck constraint (lowest efficiency).
    """
    # prizes: (5, 100)
    # weights: (5, 100)
    
    # Total prize per item (sum across the 5 prize dimensions)
    total_prizes = np.sum(prizes, axis=0)  # shape (100,)
    
    eps = 1e-9
    
    # Avoid division by zero in weights
    safe_weights = weights + eps
    
    # Efficiency per constraint: E[j, i] = total_prize[i] / weight[j, i]
    # Shape: (5, 100)
    efficiencies = total_prizes[np.newaxis, :] / safe_weights
    
    # Handle items with 0 prize: efficiency is 0.
    # We want their score to be 0 (or min positive).
    
    # Power Mean with p = -10
    # M_p = ( (1/n) * sum(x^p) )^(1/p)
    # Here n=5 (constraints)
    
    # Calculate x^p. For x=0, 0^-10 is inf.
    # We'll compute and handle inf/nan later.
    
    # To avoid warnings, we can clip efficiencies to a small positive value
    # before power, but we want 0 prize items to have 0 score.
    # If we clip E to eps, E^-10 is huge, sum is huge, M is small.
    # This effectively gives them a very low score, which is correct.
    
    eff_clipped = np.maximum(efficiencies, eps)
    
    powered = np.power(eff_clipped, -10)
    
    # Sum over constraints (axis 0)
    sum_powered = np.sum(powered, axis=0)  # shape (100,)
    
    # Mean
    mean_powered = sum_powered / 5.0
    
    # Inverse power
    scores = np.power(mean_powered, -0.1)
    
    # For items with 0 prize, efficiencies were eps, so scores will be small.
    # Let's explicitly set scores for 0-prize items to 1e-9 to be safe/clean.
    zero_prize_mask = (total_prizes < 1e-12)
    scores[zero_prize_mask] = 1e-9
    
    # Ensure positive
    scores = np.maximum(scores, 1e-9)
    
    return scores
