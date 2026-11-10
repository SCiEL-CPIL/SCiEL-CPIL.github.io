import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 75  # 15 seconds safety margin before 90s kill

    def greedy_fallback():
        """Greedy from top down, avoiding subset sum 200."""
        selected, reachable = [], {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected

    best_result = greedy_fallback()
    best_card = len(best_result)
    
    # Strategy: Find all subsets of [1,100] that sum to 200, then find a minimum hitting set
    # or equivalently, maximize the subset of [1,100] that doesn't contain any such subset.
    
    # First, let's try a different approach: use the fact that if we take all numbers > 100,
    # no two can sum to 200 (since max is 100+99=199). But three or more can.
    
    # Key insight: For numbers in [51, 100], any pair sums to at least 103, and 
    # 51+52+53 = 156, 51+52+53+54 = 210 > 200. So only triples from [51,100] can sum to 200.
    # Actually let's check: min triple from [51,100] is 51+52+53=156, max is 100+99+98=297.
    # So triples in [51,100] can sum to 200.
    
    # For numbers in [67, 100]: min triple is 67+68+69=204 > 200. So no triple from [67,100] sums to 200.
    # And no pair sums to 200 (max pair is 100+99=199).
    # So [67, 100] is entirely valid! That's 34 elements.
    
    # Can we add more? Numbers in [66, 66]: 66 alone doesn't help with pairs (66+134 out of range)
    # but 66+67+68 = 201, 66+67+67 not allowed (distinct), 66+68+66 not allowed.
    # 66+67+x = 200 => x=67, but we need distinct. 66+68+x=200 => x=66, not distinct.
    # 66+69+x=200 => x=65, not in [67,100].
    # So 66 can be added! Check: does any subset of {66} ∪ [67,100] sum to 200?
    # Pairs: 66+x=200 => x=134, no. 67+68=135, etc. Max pair in [67,100] is 199.
    # Triples: 66+a+b=200 where a,b in [67,100], a≠b. a+b=134. Min a+b=67+68=135>134. No.
    # 4+ elements: min is 66+67+68+69=270>200. No.
    # So {66} ∪ [67,100] works! That's 35 elements.
    
    # Can we add 65? 65+66+67=198, 65+66+68=199, 65+66+69=200! So 65,66,69 sum to 200.
    # If we include 65, we must exclude either 66 or 69 (or both).
    # Option A: Include 65, exclude 66. Then check if {65} ∪ [67,100] works.
    #   Pairs: 65+x=200 => x=135, no.
    #   Triples: 65+a+b=200, a,b in [67,100]. a+b=135. 67+68=135! So 65+67+68=200.
    #   So we'd need to exclude 67 or 68 too. This gets complicated.
    # Option B: Include 65, exclude 69. Check {65,66} ∪ [67,68,70,71,...,100].
    #   65+66+x=200 => x=69, excluded. Good.
    #   65+a+b=200, a,b in {66}∪[67,68,70,...,100]. If a=66, b=69 excluded. 
    #   If a,b in [67,68,70,...,100]: a+b=135. 67+68=135. So 65+67+68=200. Bad.
    #   So we'd need to exclude 67 or 68.
    
    # Let's just use ILP on the full [1,100] set but with a smarter constraint generation.
    
    # Generate all subsets of [1,100] that sum to 200 (with distinct elements)
    forbidden = []
    
    # For efficiency, use DFS with pruning
    def find_subsets_summing_to(target, elements, start_idx, current_subset, current_sum):
        if current_sum == target:
            if current_subset:
                forbidden.append(tuple(current_subset))
            return
        if current_sum > target:
            return
        if start_idx >= len(elements):
            return
        
        # Pruning: even taking all remaining min elements won't reach target
        # Or: current_sum + sum of all remaining < target
        remaining_min = elements[start_idx]
        # Upper bound on how much more we can add
        # We need to be careful - this is exponential in worst case
        
        for i in range(start_idx, len(elements)):
            if current_sum + elements[i] > target:
                break
            current_subset.append(elements[i])
            find_subsets_summing_to(target, elements, i+1, current_subset, current_sum + elements[i])
            current_subset.pop()
    
    if time.time() - start_time < timeout - 30:
        all_elements = list(range(1, 101))
        find_subsets_summing_to(200, all_elements, 0, [], 0)
    
    if time.time() - start_time > timeout:
        return best_result
    
    if not forbidden:
        return list(range(1, 101))
    
    # Now solve: maximize |S| such that S doesn't contain any forbidden subset
    # This is equivalent to: for each forbidden subset F, not all elements of F are in S
    # i.e., sum_{i in F} x_i <= |F| - 1
    
    n = 100
    c = -np.ones(n)
    
    constraints_list = []
    for f in forbidden:
        row = np.zeros(n)
        for val in f:
            row[val - 1] = 1  # 0-indexed
        constraints_list.append((row, len(f) - 1))
    
    if constraints_list:
        A_ub = np.array([r for r, _ in constraints_list])
        b_ub = np.array([b for _, b in constraints_list])
        
        integrality = np.ones(n)
        bounds = Bounds(lb=0, ub=1)
        
        try:
            res = milp(c=c, constraints=[LinearConstraint(A_ub, -np.inf, b_ub)], 
                      integrality=integrality, bounds=bounds,
                      options={"time_limit": 25})
            
            if res.status == 0:
                selected_indices = np.where(res.x > 0.5)[0]
                selected_values = [i + 1 for i in selected_indices]
                
                if len(selected_values) > best_card:
                    best_result = selected_values
                    best_card = len(selected_values)
        except Exception:
            pass
    
    # Also try the constructive approach: [66, 100] gives 35 elements
    constructive = list(range(66, 101))
    if len(constructive) > best_card:
        best_result = constructive
        best_card = len(best_result)
    
    # Try to extend [66,100] by checking if we can add smaller elements
    # We know 65 can't be added without conflicts. Let's try a greedy extension from below.
    current_set = set(constructive)
    reachable_sums = {0}
    for x in constructive:
        new_sums = {s + x for s in reachable_sums if s + x <= 200}
        reachable_sums |= new_sums
    
    # Try adding elements from 65 down to 1
    for x in range(65, 0, -1):
        if time.time() - start_time > timeout - 5:
            break
        new_sums = {s + x for s in reachable_sums if s + x <= 200}
        if 200 not in new_sums:
            current_set.add(x)
            reachable_sums |= new_sums
    
    extended = sorted(current_set)
    if len(extended) > best_card:
        best_result = extended
        best_card = len(best_result)
    
    return best_result
