"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    unit_fracs = [Fraction(1, d) for d in denominators]

    # Compute LCM of 2..48 for exact integer arithmetic
    L = 1
    for d in denominators:
        L = L * d // math.gcd(L, d)
    
    # Integer weights: weight[i] = L // denominators[i], so sum of weights / L = reciprocal sum
    weights = [L // d for d in denominators]
    target = L  # we want sum of weights strictly below L, minimize L - sum

    # Meet-in-the-middle approach
    # Split into two halves
    mid = n // 2  # 24
    left_weights = weights[:mid]
    right_weights = weights[mid:]
    n_left = len(left_weights)
    n_right = len(right_weights)

    # Generate all subset sums for left half (2^24 = 16M subsets - too many to store all)
    # Instead, let's use a different strategy:
    # Since we need sum < L and want to minimize L - sum, we want sum as close to L as possible from below.
    # 
    # Alternative: Use branch and bound with the weights sorted.
    # Or: Use a smarter meet-in-the-middle where we generate left sums and binary search right sums.
    
    # Let's try a different approach: since n=47 is moderate, use a greedy + local search 
    # but with a better initial solution and more aggressive search.
    
    # Actually, let's do a proper meet-in-the-middle with pruning.
    # Generate all subset sums for the first 23 elements (2^23 = 8M - manageable)
    # and all subset sums for the remaining 24 elements.
    
    # But storing 8M Fractions is heavy. Let's use integers.
    
    # Generate all subset sums for left part (first 23 elements)
    n_left_gen = 23
    left_gen_weights = weights[:n_left_gen]
    left_sums = []
    for mask in range(1 << n_left_gen):
        s = 0
        for i in range(n_left_gen):
            if mask & (1 << i):
                s += left_gen_weights[i]
        if s < L:  # only keep sums strictly below target
            left_sums.append(s)
    
    # Sort left sums
    left_sums.sort()
    
    # For the right part (elements 23 to 46, i.e., 24 elements), we can't enumerate all 2^24 subsets easily
    # So instead, for each left sum, we want to find the best right sum such that left_sum + right_sum < L
    # i.e., right_sum < L - left_sum, and we want right_sum as large as possible.
    
    # Generate all subset sums for right part
    right_gen_weights = weights[n_left_gen:]
    n_right_gen = len(right_gen_weights)  # 24
    right_sums = []
    for mask in range(1 << n_right_gen):
        s = 0
        for i in range(n_right_gen):
            if mask & (1 << i):
                s += right_gen_weights[i]
        if s < L:
            right_sums.append(s)
    
    right_sums.sort()
    
    # For each left sum, binary search for the largest right sum < L - left_sum
    import bisect
    best_total = 0
    best_left_mask = 0
    best_right_mask = 0
    
    # To avoid iterating over all 8M left sums, let's be smarter.
    # Actually, 8M iterations with binary search (log(16M) ~ 24) is about 200M operations, might be too slow.
    
    # Let's limit: only check top K left sums (largest ones below L)
    # Since we want left_sum + right_sum close to L, we want left_sum close to L.
    # Take the largest min(200000, len(left_sums)) left sums.
    
    max_left_to_check = min(300000, len(left_sums))
    left_to_check = left_sums[-max_left_to_check:]  # largest sums
    
    for left_sum in left_to_check:
        remaining = L - left_sum
        # Find largest right_sum < remaining
        idx = bisect.bisect_left(right_sums, remaining)
        if idx > 0:
            right_sum = right_sums[idx - 1]
            total = left_sum + right_sum
            if total > best_total:
                best_total = total
                # Recover masks
                # Find which mask gives left_sum
                # This is expensive. Let's store masks along with sums.
                # Actually, let's just do local search from a greedy solution instead.
                pass
    
    # The mask recovery is expensive. Let's use a hybrid approach:
    # 1. Get a good initial solution via greedy
    # 2. Use the meet-in-the-middle to find the theoretical best deficit
    # 3. Use local search to try to achieve it
    
    # Greedy baseline
    greedy_selected = []
    greedy_total = 0
    for i in range(n):
        if greedy_total + weights[i] < L:
            greedy_selected.append(i)
            greedy_total += weights[i]
    
    best_indices = greedy_selected[:]
    best_total = greedy_total
    best_deficit = L - best_total
    
    # Local search: 1-swap, 2-swap, add, remove
    improved = True
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        current_set = set(best_indices)
        current_total = best_total
        current_deficit = best_deficit
        
        # Try adding any element
        for i in range(n):
            if i not in current_set:
                new_total = current_total + weights[i]
                if new_total < L:
                    new_deficit = L - new_total
                    if new_deficit < current_deficit:
                        best_indices.append(i)
                        current_set.add(i)
                        current_total = new_total
                        best_total = new_total
                        best_deficit = new_deficit
                        improved = True
                        break
        
        if improved:
            continue
        
        # Try 1-swap
        for i_out in list(best_indices):
            total_without = current_total - weights[i_out]
            for i_in in range(n):
                if i_in in current_set:
                    continue
                new_total = total_without + weights[i_in]
                if new_total < L:
                    new_deficit = L - new_total
                    if new_deficit < current_deficit:
                        best_indices.remove(i_out)
                        best_indices.append(i_in)
                        current_set.remove(i_out)
                        current_set.add(i_in)
                        current_total = new_total
                        best_total = new_total
                        best_deficit = new_deficit
                        improved = True
                        break
            if improved or time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
        
        # Try 2-swap
        from itertools import combinations
        idx_list = list(best_indices)
        for i_out1, i_out2 in combinations(idx_list, 2):
            if time.time() > deadline:
                break
            total_without = current_total - weights[i_out1] - weights[i_out2]
            for i_in1 in range(n):
                if i_in1 in current_set:
                    continue
                for i_in2 in range(i_in1 + 1, n):
                    if i_in2 in current_set:
                        continue
                    new_total = total_without + weights[i_in1] + weights[i_in2]
                    if new_total < L:
                        new_deficit = L - new_total
                        if new_deficit < current_deficit:
                            best_indices.remove(i_out1)
                            best_indices.remove(i_out2)
                            best_indices.append(i_in1)
                            best_indices.append(i_in2)
                            current_set.discard(i_out1)
                            current_set.discard(i_out2)
                            current_set.add(i_in1)
                            current_set.add(i_in2)
                            current_total = new_total
                            best_total = new_total
                            best_deficit = new_deficit
                            improved = True
                            break
                if improved or time.time() > deadline:
                    break
            if improved or time.time() > deadline:
                break
    
    # Try random restarts with local search
    if time.time() < deadline - 10.0:
        import random
        random.seed(12345)
        for restart in range(2000):
            if time.time() > deadline:
                break
            # Random subset of reasonable size
            subset_size = random.randint(5, 15)
            candidate = set(random.sample(range(n), subset_size))
            total = sum(weights[i] for i in candidate)
            if total >= L:
                continue
            deficit = L - total
            if deficit >= best_deficit:
                continue
            
            # Local improvement
            changed = True
            while changed and time.time() < deadline:
                changed = False
                # Add
                for i in range(n):
                    if i in candidate:
                        continue
                    new_total = total + weights[i]
                    if new_total < L:
                        new_deficit = L - new_total
                        if new_deficit < deficit:
                            candidate.add(i)
                            total = new_total
                            deficit = new_deficit
                            changed = True
                            break
                if changed:
                    continue
                # 1-swap
                for i_out in list(candidate):
                    total_without = total - weights[i_out]
                    for i_in in range(n):
                        if i_in in candidate:
                            continue
                        new_total = total_without + weights[i_in]
                        if new_total < L:
                            new_deficit = L - new_total
                            if new_deficit < deficit:
                                candidate.discard(i_out)
                                candidate.add(i_in)
                                total = new_total
                                deficit = new_deficit
                                changed = True
                                break
                    if changed:
                        break
            
            if deficit < best_deficit:
                best_indices = list(candidate)
                best_deficit = deficit
    
    # Final validation
    final_total = sum(weights[i] for i in best_indices)
    if final_total >= L or not best_indices:
        best_indices = greedy_selected
        final_total = sum(weights[i] for i in best_indices)
    
    return best_indices
# EVOLVE-BLOCK-END
