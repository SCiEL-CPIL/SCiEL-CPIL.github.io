"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores based on competitive exclusion.
    Simulates a greedy packing process to estimate the marginal utility
    of each item relative to the capacity consumed by higher-prize items.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit

    n_constraints, n_items = prizes.shape
    capacity = 25.0
    
    # Sort items by prize in descending order to determine priority
    # Higher prize items are considered "prior" in the competition
    order = np.argsort(-prizes.sum(axis=0))
    
    # Initialize scores
    scores = np.zeros(n_items)
    
    # Track remaining capacity for the greedy simulation
    # We simulate adding items one by one in order of prize
    # If an item fits, it "excludes" future items from using that capacity
    # The heuristic score for an item is its prize divided by the "pressure"
    # exerted on its constraints by higher-priority items.
    
    # For efficiency, we can approximate the "pressure" by tracking
    # how much capacity is left for each constraint as we go down the sorted list.
    # However, a true competitive exclusion score might be:
    # score[i] = prize[i] / product(1 + sum_{j < i, j competes with i} weight[j,c])
    
    # Let's use a simpler, more direct competitive metric:
    # For each item i, calculate the total weight of all items j with higher prize
    # that share any constraint with i (which is all of them since all items have weights in all constraints).
    # Actually, the constraint is per dimension.
    # Let's define "exclusion pressure" for item i in constraint c as the sum of weights
    # of all items j with prize > prize_i that are "selected" in a greedy pass.
    
    # A more robust approach:
    # 1. Sort items by prize descending.
    # 2. Iterate through sorted items.
    # 3. Maintain a vector of 'used capacity' for each constraint.
    # 4. For each item, if it fits, mark it as 'selected' and update used capacity.
    # 5. The heuristic score for a selected item is high (it was competitive enough to be picked).
    # 6. The heuristic score for a non-selected item is lower, proportional to how close it was to fitting.
    
    # But the ACO uses these scores for ALL items. Items that don't fit later are still candidates
    # in the ACO if the ant's path is different. So we need a score that reflects potential value
    # relative to competition.
    
    # Alternative Strategy: 
    # Score[i] = Prize[i] / (1 + CompetitorWeight[i])
    # Where CompetitorWeight[i] is the sum of weights of items with higher prize in the bottleneck constraint.
    
    # Let's try: 
    # 1. Identify the bottleneck constraint for the whole problem? No, it's per item.
    # 2. For each item, look at the constraint where it is "most expensive" relative to the average.
    
    # Let's stick to the "Marginal Utility" idea.
    # Simulate a greedy fill.
    # remaining_cap = np.full(n_constraints, capacity)
    # selected = np.zeros(n_items, dtype=bool)
    
    # For each item in order:
    #   if it fits in remaining_cap:
    #       selected[i] = True
    #       remaining_cap -= weights[:, i]
    #       score[i] = 1.0  # Or some high value
    #   else:
    #       # It didn't fit. How much did it miss by?
    #       # The "pressure" is high.
    #       # score[i] = prize[i] / (sum of weights of selected items in the blocking constraint)
    
    # This is O(N^2) if we recompute, but we can do it in O(N log N + N*C).
    
    # Let's compute the score as:
    # score[i] = prize[i] / (1 + max_c (sum_{j: prize[j]>prize[i]} weight[j,c] * indicator(j selected)))
    
    # This requires knowing which j are selected.
    
    # Let's implement the greedy pass to determine 'selected' and 'remaining_cap' at the point of consideration.
    
    remaining_cap = np.full(n_constraints, capacity)
    # We need to assign a score to each item.
    # If an item is selected in the greedy pass, it has proven high utility.
    # If not, it was blocked.
    
    # Let's assign:
    # If selected: score = 1.0 + prize[i]
    # If not selected: score = prize[i] / (1 + sum of weights of selected items in the tightest constraint)
    
    # Wait, the ACO heuristic is multiplied by pheromone. 
    # If we set high scores for greedy-selected items, the ACO will likely re-select them.
    # This might be good (exploitation) but bad (exploration).
    # However, the task asks for "attractiveness".
    
    # Let's try a different angle: 
    # The "competitive exclusion" score is the ratio of the item's prize to the total prize of items 
    # that are "more efficient" and compete for the same resources.
    
    # Efficient items are those with high prize/weight.
    # Let's rank items by prize/weight.
    # For item i, let S_i be the set of items j such that prize[j]/weight[j] > prize[i]/weight[i].
    # The exclusion score is prize[i] / (1 + sum_{j in S_i} weight[j,c_max])
    # where c_max is the constraint where item i has the highest relative weight.
    
    # This is complex. Let's use a simpler proxy:
    # Score[i] = prize[i] / (1 + sum_{j: prize[j] > prize[i]} weights[j, c_i])
    # where c_i is the constraint where item i is most "constrained" (highest weight relative to capacity).
    
    # Let's compute c_i first.
    # relative_weight[i, c] = weights[i, c] / capacity
    # c_i = argmax_c relative_weight[i, c]
    
    rel_weights = weights / capacity
    c_i = np.argmax(rel_weights, axis=0)
    
    # Now, for each item i, we need the sum of weights in constraint c_i 
    # for all items j with higher prize.
    
    # We can sort items by prize.
    # Let sorted_indices = np.argsort(-prizes.sum(axis=0))
    # Let sorted_prizes = prizes.sum(axis=0)[sorted_indices]
    # Let sorted_weights = weights[:, sorted_indices]
    
    # For the k-th item in this sorted order (item i_k), 
    # the "competitors" are items i_0 ... i_{k-1}.
    # We need sum_{m=0}^{k-1} weights[c_i_k, i_m]
    
    # This is a cumulative sum problem, but the constraint index c_i_k varies.
    
    # We can maintain a running sum of weights for each constraint.
    # running_sums[c] = sum of weights[c, i_m] for m < k
    
    sorted_indices = np.argsort(-prizes.sum(axis=0))
    sorted_prizes = prizes.sum(axis=0)[sorted_indices]
    
    scores = np.zeros(n_items)
    
    running_sums = np.zeros(n_constraints)
    
    for k in range(n_items):
        if time.time() > deadline:
            break
            
        i = sorted_indices[k]
        c = c_i[i]
        
        # The exclusion pressure is the running sum of weights in constraint c 
        # from all higher-prize items.
        pressure = running_sums[c]
        
        # Score is prize relative to this pressure.
        # If pressure is high, the item is heavily excluded.
        # If pressure is low, it has high potential.
        # We want larger scores to be more attractive.
        # So score = prize / (1 + pressure)
        
        scores[i] = prizes.sum(axis=0)[i] / (1.0 + pressure)
        
        # Update running sums with the current item's weights
        # Only if it would have been "selected" in a greedy sense?
        # The strategy says "relative to the sum of prizes of all other items that would violate... if selected"
        # Actually, "items that would violate... if selected" implies items that are *already* selected.
        # In a greedy pass, an item is selected if it fits.
        
        # Let's refine: Only update running_sums if the item fits in the remaining capacity 
        # of the greedy simulation. This models the actual exclusion.
        
        # We need to track remaining capacity for the greedy simulation too.
        # But we already have running_sums which is the sum of weights of *selected* items.
        # So remaining_cap_greedy = capacity - running_sums
        
        if np.all(weights[:, i] <= capacity - running_sums + 1e-9):
            # Item fits, so it is selected and excludes others
            running_sums += weights[:, i]
        # else: item is not selected, so it doesn't add to the exclusion pressure for subsequent items
        # Wait, if it's not selected, it doesn't consume capacity, so it doesn't exclude others.
        # But its score was calculated based on the pressure from *previous* selected items.
        # This seems consistent.
        
    # Ensure positive scores
    scores = np.maximum(scores, 1e-9)
    
    return scores

# EVOLVE-BLOCK-END
