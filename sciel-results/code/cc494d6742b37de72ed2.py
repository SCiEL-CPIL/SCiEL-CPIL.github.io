"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack.
    
    Strategy: For each constraint, compute global scarcity (total weight / capacity).
    An item's effective cost is the sum over constraints of (weight_in_c * scarcity_c).
    Heuristic = prize / (effective_cost + eps), which rewards items that are
    efficient relative to the scarcity of the resources they consume.
    """
    n_items = weights.shape[1]
    capacity = 25.0
    
    # Global scarcity per constraint: how much total demand exceeds capacity
    # scarcity_c = sum_i weights[c,i] / capacity
    scarcity = weights.sum(axis=1) / capacity  # shape (5,)
    
    # Effective cost per item: sum over constraints of weight[c,i] * scarcity[c]
    # This penalizes items that use scarce resources more heavily
    effective_cost = (weights * scarcity[:, np.newaxis]).sum(axis=0)  # shape (100,)
    
    # Prize per item (sum over constraints? or use mean? Let's use sum of prizes)
    # Actually prizes shape is (5, 100). Each item has a prize per constraint? 
    # Re-reading: "prizes and weights, of shape (5, 100)". 
    # The total prize of an item is likely the sum or some aggregate.
    # In knapsack, each item has one prize. But here shape is (5,100).
    # Looking at the seed: prizes / (weights.sum(axis=0) + 1e-9)
    # This divides prize by sum of weights. So prizes is (5,100) and they're dividing element-wise?
    # No, weights.sum(axis=0) is (100,), prizes is (5,100). That would be broadcasting error.
    # Wait, let me re-read. The seed does: prizes / (weights.sum(axis=0) + 1e-9)
    # If prizes is (5,100) and weights.sum(axis=0) is (100,), this broadcasts to (5,100)/(100,) = (5,100)
    # But the return should be "one score per item", shape (100,).
    # So maybe prizes is actually (100,) in practice? Or the seed is wrong?
    # Let me assume prizes is (100,) for the return to be (100,).
    # Actually the contract says "prizes and weights, of shape (5, 100)". 
    # But the seed returns prizes / (weights.sum(axis=0) + 1e-9) which would be (5,100) if prizes is (5,100).
    # That doesn't match "one score per item". 
    # I'll assume prizes is (100,) - the total prize per item. Or maybe I should sum prizes.
    # Let me just use prizes.sum(axis=0) to get one prize per item, to be safe.
    
    prize_per_item = prizes.sum(axis=0)  # shape (100,)
    
    # Heuristic: prize / effective_cost
    heuristic = prize_per_item / (effective_cost + 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
