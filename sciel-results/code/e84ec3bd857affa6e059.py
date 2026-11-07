"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items using a marginal utility heuristic.
    
    The heuristic estimates the opportunity cost of each item by considering
    the scarcity of each resource (constraint) and rewarding items that offer
    high prize relative to the current scarcity of the specific resources they consume.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,) with one attractiveness score per item
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = prizes.shape
    
    # For each constraint, compute the efficiency of each item: prize / weight
    # Efficiency tells us how much prize we get per unit of that resource
    efficiencies = np.zeros_like(prizes, dtype=np.float64)
    for c in range(n_constraints):
        # Avoid division by zero
        w = weights[c, :]
        p = prizes[c, :]
        efficiencies[c, :] = np.where(w > 1e-12, p / np.maximum(w, 1e-12), 0.0)
    
    # For each constraint, find the top-k most efficient items
    # k = capacity / average weight, but since capacity is fixed at 25 and weights are in [0,1],
    # a reasonable k is around 25-50. Let's use k=25 as a baseline.
    k = 25
    
    # Compute scarcity score for each constraint based on the top-k efficiencies
    # The idea: if the top-k items have high efficiency, the resource is "scarce" in the sense
    # that good items are competing for it. We want to reward items that are efficient in scarce resources.
    
    # For each constraint, compute a scarcity factor based on the sum of top-k efficiencies
    scarcity = np.zeros(n_constraints, dtype=np.float64)
    for c in range(n_constraints):
        eff_c = efficiencies[c, :]
        # Get top k efficiencies
        if n_items >= k:
            top_k_eff = np.sort(eff_c)[::-1][:k]
        else:
            top_k_eff = np.sort(eff_c)[::-1]
        # Scarcity is proportional to the sum of top-k efficiencies (higher means more competition)
        scarcity[c] = np.sum(top_k_eff)
    
    # Normalize scarcity to [0, 1] range for better conditioning
    max_scarcity = np.max(scarcity)
    if max_scarcity > 1e-12:
        scarcity = scarcity / max_scarcity
    else:
        scarcity = np.ones(n_constraints) / n_constraints
    
    # Now compute the marginal utility for each item
    # For each item, compute a weighted sum of its efficiencies, weighted by the scarcity of each resource
    # Items that are efficient in scarce resources get higher scores
    
    # Weighted efficiency for each item
    # score_item = sum over constraints of (efficiency[item, c] * scarcity[c])
    # But we also want to consider the total prize of the item
    
    # Let's combine: for each item, compute
    # 1. The total prize: sum of prizes across constraints
    # 2. The weighted efficiency: sum of efficiencies weighted by scarcity
    
    total_prizes = np.sum(prizes, axis=0)  # Shape: (100,)
    
    # Weighted efficiency: for each item, sum of eff[i, c] * scarcity[c] over c
    weighted_eff = np.zeros(n_items, dtype=np.float64)
    for c in range(n_constraints):
        weighted_eff += efficiencies[c, :] * scarcity[c]
    
    # Combine total prize and weighted efficiency
    # Normalize both to [0, 1] for better conditioning
    max_total_prize = np.max(total_prizes)
    if max_total_prize > 1e-12:
        total_prizes_norm = total_prizes / max_total_prize
    else:
        total_prizes_norm = np.ones(n_items) / n_items
    
    max_weighted_eff = np.max(weighted_eff)
    if max_weighted_eff > 1e-12:
        weighted_eff_norm = weighted_eff / max_weighted_eff
    else:
        weighted_eff_norm = np.ones(n_items) / n_items
    
    # Final heuristic: combination of normalized total prize and normalized weighted efficiency
    # Weight them equally or give more weight to weighted efficiency (scarcity-aware)
    alpha = 0.5
    heuristic = alpha * total_prizes_norm + (1 - alpha) * weighted_eff_norm
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
