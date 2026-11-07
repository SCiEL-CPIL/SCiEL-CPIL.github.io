"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute dynamic 'marginal utility' heuristic scores for each item.
    
    The heuristic estimates opportunity cost by considering:
    1. Efficiency of each item (prize / resource consumption) per constraint
    2. Scarcity of each resource based on how efficiently other items use it
    3. Reward items that offer high prize relative to the current scarcity of 
       the specific resources they consume
    
    Args:
        prizes: array of shape (5, 100) - prize values for 5 constraints, 100 items
        weights: array of shape (5, 100) - weight/resource consumption for 5 constraints, 100 items
    
    Returns:
        array of shape (100,) - one attractiveness score per item (larger is more attractive)
    """
    # Ensure inputs are numpy arrays with correct shape
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("prizes and weights must have shape (5, 100)")
    
    n_constraints, n_items = prizes.shape
    
    # Clip weights to avoid division by zero, but keep small values meaningful
    weights_safe = np.maximum(weights, 1e-9)
    
    # Compute efficiency per constraint: prize / weight for each item-constraint pair
    # efficiency[c, i] = prizes[c, i] / weights[c, i]
    efficiency = prizes / weights_safe  # shape (5, 100)
    
    # For each constraint, compute the scarcity factor based on top-k most efficient items
    # Idea: if many items are very efficient at using a resource, that resource is "scarce"
    # in the sense that competition for it is high, so items using it should be penalized
    # unless they are among the most efficient.
    
    # Alternative interpretation: scarcity = how much total prize is "competing" for each unit
    # of resource. We use the sum of top-k efficiencies as a proxy for scarcity pressure.
    
    k = 10  # top-k items to consider for scarcity estimation
    
    # For each constraint, get the top-k efficiencies and their sum
    scarcity_pressure = np.zeros(n_constraints)
    for c in range(n_constraints):
        # Get the k-th largest efficiency (or min if fewer than k items)
        k_eff = np.sort(efficiency[c, :])[-min(k, n_items):]
        scarcity_pressure[c] = np.sum(k_eff)
    
    # Normalize scarcity pressure so that average is 1
    mean_scarcity = np.mean(scarcity_pressure)
    if mean_scarcity > 0:
        scarcity_pressure = scarcity_pressure / mean_scarcity
    else:
        scarcity_pressure = np.ones(n_constraints)
    
    # Now compute a weighted efficiency for each item, where the weight of each constraint's
    # efficiency is inversely proportional to the scarcity pressure of that constraint.
    # This rewards items that are efficient in less-scarce resources.
    
    # weight_for_constraint[c] = 1 / scarcity_pressure[c]
    constraint_weights = 1.0 / scarcity_pressure  # shape (5,)
    
    # Normalize constraint weights to sum to 1
    total_cw = np.sum(constraint_weights)
    if total_cw > 0:
        constraint_weights = constraint_weights / total_cw
    else:
        constraint_weights = np.ones(n_constraints) / n_constraints
    
    # Compute weighted efficiency for each item
    # item_score[i] = sum_c(constraint_weights[c] * efficiency[c, i])
    item_scores = constraint_weights @ efficiency  # shape (100,)
    
    # Additionally, consider the total prize of each item relative to total resource consumption
    total_prize = np.sum(prizes, axis=0)  # shape (100,)
    total_weight = np.sum(weights_safe, axis=0)  # shape (100,)
    overall_efficiency = total_prize / (total_weight + 1e-9)
    
    # Combine: use a blend of the scarcity-aware weighted efficiency and overall efficiency
    # Normalize both to have similar scales
    if np.max(item_scores) > 0:
        item_scores_norm = item_scores / np.max(item_scores)
    else:
        item_scores_norm = np.zeros(n_items)
    
    if np.max(overall_efficiency) > 0:
        overall_eff_norm = overall_efficiency / np.max(overall_efficiency)
    else:
        overall_eff_norm = np.zeros(n_items)
    
    # Blend with 60% scarcity-aware, 40% overall
    combined = 0.6 * item_scores_norm + 0.4 * overall_eff_norm
    
    # Ensure all values are positive and finite
    combined = np.maximum(combined, 1e-9)
    combined = np.where(np.isfinite(combined), combined, 1e-9)
    
    return combined
# EVOLVE-BLOCK-END
