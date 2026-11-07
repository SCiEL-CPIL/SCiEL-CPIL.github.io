# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items using a dynamic marginal utility heuristic.
    
    The heuristic estimates the opportunity cost of each item by considering
    the scarcity of each resource (constraint). Items that offer high prize
    relative to the scarcity of the specific resources they consume are rewarded.
    
    Args:
        prizes: array of shape (5, 100) with prize values in [0, 1]
        weights: array of shape (5, 100) with weight values in [0, 1]
    
    Returns:
        array of shape (100,) with one attractiveness score per item
    """
    n_constraints, n_items = prizes.shape
    
    # Compute total prize for each item (sum across constraints? No, prizes is per-constraint)
    # Actually, looking at the task: "prizes and weights, of shape (5, 100)"
    # The total prize of an item is likely the sum or we need to think about what "prize" means.
    # In the ant colony, "best prize of that iteration" - so each item has a prize.
    # Let's assume the prize of an item is the sum of its prizes across constraints, 
    # or perhaps each constraint has its own prize vector. 
    # Re-reading: "Item scores for ant colony optimization of the multiple-constraint knapsack"
    # "prizes and weights, of shape (5, 100)" - so for each constraint c and item i, 
    # there is a prize[c, i] and weight[c, i].
    # The total prize of selecting item i would be sum_c prizes[c, i].
    
    item_prizes = prizes.sum(axis=0)  # shape (100,)
    
    # For each constraint, compute efficiency: prize contribution per unit weight
    # efficiency[c, i] = prizes[c, i] / (weights[c, i] + eps)
    eps = 1e-9
    efficiency = prizes / (weights + eps)  # shape (5, 100)
    
    # For each constraint, find the top-k most efficient items to estimate scarcity
    # We'll use k = 20 (arbitrary, but reasonable)
    k = min(20, n_items)
    
    # For each constraint, get the top-k efficiencies and their sum
    # This represents the "value" of the best items for that resource
    top_k_efficiencies = np.zeros(n_constraints)
    for c in range(n_constraints):
        top_k_vals = np.sort(efficiency[c, :])[::-1][:k]
        top_k_efficiencies[c] = top_k_vals.mean()
    
    # Now, for each item, compute a marginal utility score
    # The idea: an item is more attractive if it has high prize relative to 
    # the scarcity (high top-k efficiency) of the resources it consumes
    
    # For each item i, compute a weighted sum of the scarcity of resources it uses
    # weighted by how much of each resource it consumes
    scores = np.zeros(n_items)
    for i in range(n_items):
        # Weight vector for this item: how much of each resource it uses
        w_i = weights[:, i]  # shape (5,)
        p_i = prizes[:, i]  # shape (5,)
        
        # Opportunity cost: for each constraint, the cost is proportional to
        # how much weight we use times the scarcity (top-k efficiency) of that constraint
        # Higher top-k efficiency means the resource is more "valuable" / scarce
        # So consuming more of a scarce resource is more costly
        
        # Marginal utility = total prize - sum of opportunity costs
        # Or: prize weighted by inverse scarcity
        
        # Let's define: for each constraint c, the "cost" of using weight[c, i] is
        # weight[c, i] * top_k_efficiencies[c] * some_scale
        # And the "benefit" is prizes[c, i]
        
        # A cleaner approach: normalize efficiency per constraint
        # scarcity[c] = top_k_efficiencies[c] / (top_k_efficiencies.mean() + eps)
        # This makes scarcity dimensionless, centered around 1
        
        # Actually, let's think differently. The heuristic should be:
        # score[i] = sum_c prizes[c, i] / (1 + sum_c weights[c, i] * scarcity[c])
        # where scarcity[c] reflects how "tight" constraint c is
        
        # Compute scarcity as the ratio of top-k efficiency to mean efficiency
        mean_eff = efficiency.mean(axis=1)  # shape (5,)
        scarcity = top_k_efficiencies / (mean_eff + eps)  # shape (5,)
        # scarcity[c] > 1 means constraint c has higher-than-average top items,
        # suggesting it's more competitive/scarce
        
        # For item i, the "effective cost" is weighted by scarcity
        effective_cost = np.sum(w_i * scarcity)
        total_prize = np.sum(p_i)
        
        scores[i] = total_prize / (1.0 + effective_cost)
    
    # Ensure all values are positive
    scores = np.maximum(scores, eps)
    
    return scores
# EVOLVE-BLOCK-END
