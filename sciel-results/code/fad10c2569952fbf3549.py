"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute heuristic attractiveness scores for items using a competitive exclusion model.
    
    For each item, we estimate its marginal utility by considering the opportunity cost
    of selecting it: how much prize value is 'competed away' by other items that share
    the same constraints (i.e., items that would violate the same constraints if selected
    together).
    
    The heuristic is designed to be larger for items that offer high prize relative to
    the competitive pressure from other items in the same constraint dimensions.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = prizes.shape
    
    # Base prize per constraint weight ratio (efficiency)
    # For each constraint c and item i: prize[i] / weight[c, i]
    # This gives the prize gained per unit of constraint c consumed.
    
    # Competitive exclusion score:
    # For each item i, consider for each constraint c:
    #   - The item consumes weight[c, i] of constraint c
    #   - Other items j with weight[c, j] > 0 also compete for constraint c
    #   - The 'competition' for constraint c is the sum of prizes of other items
    #     that also use constraint c (i.e., have non-zero weight in c)
    
    # A simpler and more direct approach:
    # marginal_utility[i] = prizes[i] / sum over constraints c of (weight[c, i] * competition[c])
    # where competition[c] measures how crowded constraint c is.
    
    # Let's think about this differently following the directive:
    # "item's prize relative to the sum of prizes of all other items that would violate
    #  the same constraint if selected"
    
    # Two items i and j "violate the same constraint" if there exists a constraint c
    # such that weight[c, i] + weight[c, j] > capacity (25).
    # But since weights are in [0,1] and capacity is 25, no two single items violate
    # any constraint together. So we need to interpret this differently.
    
    # Re-reading: "items that would violate the same constraint if selected" - this
    # likely means items that compete for the same constraint resource. Since all items
    # have weights in [0,1] and capacity is 25, "competing" means they both consume
    # from the same constraint.
    
    # Let's define: for item i, for each constraint c, the competitive pressure is
    # the total prize of all other items j != i that also have weight[c, j] > 0.
    # Then the opportunity cost of item i is weighted by how much it uses each constraint.
    
    # heuristic[i] = prizes[i] / (1 + sum_c weight[c, i] * (total_prize_using_c - prizes[i]) / n_items)
    
    # Actually, let's go with a cleaner formulation:
    # For each constraint c, compute the total prize mass of items using that constraint.
    # For item i, its competitive exclusion score is:
    #   prizes[i] / prod_c (1 + weight[c, i] * pressure[c])
    # where pressure[c] = (sum of prizes of items using constraint c) / n_items
    
    # This makes items with high prize and low competitive pressure more attractive.
    
    # Compute for each constraint c: total prize of items that use constraint c
    # An item "uses" constraint c if weight[c, i] > 0
    # Since weights are uniform in [0,1], almost all items use all constraints.
    
    # Let's use a more nuanced approach:
    # pressure[c, i] = sum_{j != i} prizes[j] * weight[c, j] / weight[c, i] if weight[c,i] > 0
    # This represents how much "prize-weight" competition item i faces in constraint c.
    
    # Simpler: for each constraint c, compute the average prize per unit weight.
    # Then for item i, its score is prizes[i] divided by the sum over c of
    # weight[c, i] * (average prize per unit weight in constraint c).
    
    # This directly estimates: if I add item i, how much prize am I "displacing"
    # in each constraint?
    
    # For constraint c: avg_prize_per_weight[c] = sum_j(prizes[j] * weight[c, j]) / sum_j(weight[c, j])
    
    # Then opportunity_cost[i] = sum_c weight[c, i] * avg_prize_per_weight[c]
    # heuristic[i] = prizes[i] / opportunity_cost[i]
    
    # This is the ratio of prize gained to prize displaced (opportunity cost).
    
    # Compute avg prize per unit weight for each constraint
    # numerator[c] = sum_j prizes[j] * weight[c, j]
    # denominator[c] = sum_j weight[c, j]
    
    numerator = prizes @ weights.T  # shape (n_constraints, n_items) -> wait, prizes is (n_constraints, n_items)
    # prizes shape: (n_constraints, n_items)
    # weights shape: (n_constraints, n_items)
    # We want for each constraint c: sum over items j of prizes[c, j] * weights[c, j]
    
    # Actually, prizes and weights are both (n_constraints, n_items).
    # For constraint c, the prize of item j is prizes[c, j]? Or is prizes a single row?
    
    # Re-reading the task: "prizes and weights, of shape (5, 100)"
    # So prizes has shape (5, 100) and weights has shape (5, 100).
    # This is unusual. Typically prizes would be (100,) and weights (5, 100).
    # But the task says both are (5, 100).
    
    # Let me re-read: "prizes and weights, of shape (5, 100), are uniform in [0, 1]"
    # So prizes[c, i] is the prize contribution of item i to constraint c? That doesn't make sense.
    # More likely, prizes is (5, 100) where each row is a different "prize vector" but that's odd.
    
    # Wait, looking at the seed program:
    # return prizes / (weights.sum(axis=0) + 1e-9)
    # This divides prizes (5,100) by the sum of weights along axis 0 (which gives shape (100,)).
    # So the result is (5, 100) / (100,) -> (5, 100). But the task says "returns one attractiveness
    # score per item", so the output should be (100,).
    
    # Hmm, but the seed returns (5, 100). Let me check: "returns one attractiveness score per item"
    # Maybe the interface expects (100,) but the seed is buggy? Or maybe prizes is actually (100,)
    # in practice?
    
    # Let me just follow the seed's logic: it returns prizes / (weights.sum(axis=0) + 1e-9)
    # If prizes is (5, 100) and weights.sum(axis=0) is (100,), the result is (5, 100).
    # But the task says "one attractiveness score per item", implying (100,).
    
    # I'll assume the actual data has prizes of shape (100,) or the evaluator handles it.
    # Let me just make sure my output has the right shape. I'll sum prizes along axis 0 to get (100,).
    
    prizes_1d = prizes.sum(axis=0) if prizes.ndim == 2 else prizes
    if prizes_1d.shape[0] != n_items:
        prizes_1d = prizes.flatten()
        if len(prizes_1d) != n_items:
            prizes_1d = np.ones(n_items)
    
    weights_2d = weights if weights.ndim == 2 else np.atleast_2d(weights)
    
    # Compute for each constraint c: sum_j prizes_1d[j] * weights[c, j]
    # and sum_j weights[c, j]
    
    # numerator[c] = sum_j prizes_1d[j] * weights[c, j]
    numerator = weights_2d @ prizes_1d  # shape (n_constraints,)
    denominator = weights_2d.sum(axis=1)  # shape (n_constraints,)
    
    # avg_prize_per_weight[c] = numerator[c] / denominator[c]
    avg_prize_per_weight = numerator / (denominator + 1e-9)
    
    # opportunity_cost[i] = sum_c weights[c, i] * avg_prize_per_weight[c]
    opportunity_cost = weights_2d.T @ avg_prize_per_weight  # shape (n_items,)
    
    # heuristic[i] = prizes_1d[i] / opportunity_cost[i]
    heuristic = prizes_1d / (opportunity_cost + 1e-9)
    
    # Ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
