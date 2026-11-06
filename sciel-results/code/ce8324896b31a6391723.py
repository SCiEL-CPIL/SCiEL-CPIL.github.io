"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import gcd
    
    # Set a safety deadline. The evaluator kills at 90s, but we want to return well before.
    deadline = time.time() + 55.0
    
    nums = list(range(2, 257))
    n = len(nums)
    
    # Precompute conflicts. 
    # conflict[i] is a set of indices j such that lcm(nums[i], nums[j]) <= 256.
    # Using sets for fast intersection/union checks.
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(nums[i], nums[j])
            if (nums[i] // g) * nums[j] <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    def is_valid_add(selected, i):
        """Check if adding i to selected is valid (no conflicts)."""
        # If any element in selected conflicts with i, it's invalid.
        # Since conflict[i] contains all j that conflict with i, we just check intersection.
        # However, iterating over the smaller set is faster.
        if len(selected) < len(conflict[i]):
            return not selected.intersection(conflict[i])
        else:
            return not conflict[i].intersection(selected)
    
    def local_search(init_set, max_iters=20):
        """
        Local search starting from init_set.
        Returns (best_set, best_score).
        """
        best = set(init_set)
        best_score = sum(1.0 / nums[i] for i in best)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            improved = False
            
            # 1. Try adding one element
            # We want to add the element with the highest 1/nums[i] that doesn't conflict.
            # Sorting candidates by value descending helps find the best add quickly if we just take the first valid one.
            # But to be safe, we scan all.
            best_add = -1
            best_add_val = 0.0
            for i in range(n):
                if i in best:
                    continue
                if is_valid_add(best, i):
                    val = 1.0 / nums[i]
                    if val > best_add_val:
                        best_add_val = val
                        best_add = i
            
            if best_add != -1:
                best.add(best_add)
                best_score += best_add_val
                improved = True
            
            # 2. Try remove-one-add-one (swap)
            # This is more expensive. We try to remove a high-cost element (low reciprocal) 
            # and add a low-cost element (high reciprocal) if possible.
            # Actually, we want to maximize sum. 
            # Swap (i in best, j not in best) is beneficial if 1/nums[j] > 1/nums[i] 
            # AND j is compatible with best\{i}.
            
            # Optimization: Only consider swaps if we have time.
            if time.time() < deadline - 5.0:
                best_swap = None
                best_swap_gain = 0.0
                
                # Iterate over elements in best
                # To speed up, we might just try a few random swaps or all if set is small.
                # Given N=255, O(N^2) is ~65k ops. In Python, this might be ~0.05-0.1s per iteration.
                # We have ~20 iterations. That's 1-2s. Acceptable.
                
                for i in best:
                    if time.time() > deadline:
                        break
                    # Calculate score if we remove i
                    # We don't need to recalculate full score, just gain
                    # Gain = 1/nums[j] - 1/nums[i]
                    # We need j not in best, and j compatible with best - {i}
                    # Let's find the best j for this specific i? No, global best swap is better.
                    # But checking all j for all i is O(N^2).
                    
                    # Heuristic: Only try swapping out larger numbers (smaller reciprocals) 
                    # for smaller numbers (larger reciprocals) that are currently excluded.
                    # If nums[i] is small (large reciprocal), it's hard to beat.
                    
                    for j in range(n):
                        if j in best:
                            continue
                        # If 1/nums[j] <= 1/nums[i], no point swapping (gain <= 0)
                        if nums[j] >= nums[i]:
                            continue
                        
                        # Check if j is compatible with best \ {i}
                        # j must not conflict with any k in best where k != i
                        # Equivalent: j not in conflict[k] for all k in best\{i}
                        # Or: (conflict[j] & (best - {i})) is empty
                        
                        # Fast check:
                        # If i is in conflict[j], that's fine because we are removing i.
                        # We need to check if conflict[j] intersects best, excluding i.
                        
                        # Optimization: 
                        # If len(conflict[j]) is small, check intersection with best.
                        # If len(best) is small, iterate best.
                        
                        # Let's use a simple set difference intersection
                        # temp_best = best - {i}
                        # if not conflict[j].intersection(temp_best):
                        
                        # To avoid creating new sets, we can do:
                        # If i in conflict[j]:
                        #    We need to ensure no OTHER element in best conflicts with j.
                        # Else:
                        #    We need to ensure NO element in best conflicts with j.
                        
                        # This is effectively: (conflict[j] & best) <= {i}
                        
                        # Let's just do the intersection check, it's C-level fast for sets.
                        # Create a view or just check.
                        
                        # If i is not in conflict[j], then conflict[j] & best must be empty.
                        # If i is in conflict[j], then (conflict[j] & best) must be exactly {i} or empty? 
                        # No, if k in best and k != i, and k in conflict[j], it's invalid.
                        # So we need: (conflict[j] & best) is subset of {i}.
                        
                        # Implementation:
                        # If i not in conflict[j]:
                        #    if conflict[j] & best: continue
                        # Else:
                        #    # i is a conflict. Check if any other element in best is a conflict.
                        #    # We can iterate best and skip i? Or just intersect and check.
                        #    # Intersect is faster.
                        #    if len(conflict[j] & best) > 1: continue 
                        #    # Wait, if len is 1, it must be i. If len is 0, it's 0.
                        #    # If len > 1, invalid.
                        #    # Actually, if i in conflict[j], then i is in the intersection.
                        #    # So we just need to check if there are OTHERS.
                        #    # i.e., len(conflict[j] & best) > 1 implies invalid.
                        #    # If len == 1, it's just i, so valid (since we remove i).
                        #    # If len == 0, it's 0, so valid.
                        
                        # Let's refine:
                        # Intersect = conflict[j] & best
                        # If i in Intersect:
                        #     If len(Intersect) > 1: Invalid
                        #     Else: Valid
                        # Else:
                        #     If len(Intersect) > 0: Invalid
                        #     Else: Valid
                        
                        inter = conflict[j] & best
                        if i in inter:
                            if len(inter) > 1:
                                continue
                        else:
                            if len(inter) > 0:
                                continue
                                
                        gain = 1.0 / nums[j] - 1.0 / nums[i]
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = (i, j)
                            
                if best_swap is not None:
                    i, j = best_swap
                    best.remove(i)
                    best.add(j)
                    best_score += best_swap_gain
                    improved = True
            
            if not improved:
                break
        
        return best, best_score

    # Initial Greedy
    # Sort by value descending (smallest numbers first) to maximize sum quickly
    order = sorted(range(n), key=lambda i: nums[i])
    initial_set = set()
    for i in order:
        if is_valid_add(initial_set, i):
            initial_set.add(i)
            
    best_set = initial_set
    best_score = sum(1.0 / nums[i] for i in best_set)
    
    # Local Search
    ls_set, ls_score = local_search(initial_set, max_iters=10)
    if ls_score > best_score:
        best_set = ls_set
        best_score = ls_score
        
    # Random Restarts
    random.seed(42)
    # Limit restarts to ensure we finish in time
    max_restarts = 50
    for k in range(max_restarts):
        if time.time() > deadline - 2.0:
            break
            
        # Generate a random start
        # Mix of random and perturbed best
        if k % 3 == 0:
            # Random subset
            s = set(random.sample(range(n), 50))
            # Filter valid ones
            valid_s = set()
            # Sort by value to prioritize high value
            for i in sorted(s, key=lambda x: nums[x]):
                if is_valid_add(valid_s, i):
                    valid_s.add(i)
            current_set = valid_s
        elif k % 3 == 1:
            # Perturb best
            current_set = set(best_set)
            # Remove a few random elements
            for _ in range(5):
                if current_set:
                    victim = random.choice(list(current_set))
                    current_set.remove(victim)
        else:
            # Greedy with random tie-breaking
            current_set = set()
            rand_order = list(range(n))
            random.shuffle(rand_order)
            # Sort by value but with random noise
            rand_order.sort(key=lambda i: nums[i] + random.random() * 10)
            for i in rand_order:
                if is_valid_add(current_set, i):
                    current_set.add(i)
        
        ls_set, ls_score = local_search(current_set, max_iters=5)
        if ls_score > best_score:
            best_set = ls_set
            best_score = ls_score
            
    return [nums[i] for i in best_set]
# EVOLVE-BLOCK-END
