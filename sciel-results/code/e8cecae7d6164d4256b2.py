import time
from math import lcm
from fractions import Fraction
import bisect
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Target is 1, which is L in integer units
    # We want max sum < L

    weights = [L // d for d in denominators]
    
    # Split into two halves for meet-in-the-middle
    mid = n // 2
    indices_1 = list(range(mid))
    indices_2 = list(range(mid, n))
    
    weights_1 = [weights[i] for i in indices_1]
    weights_2 = [weights[i] for i in indices_2]
    
    def gen_sums(weights_list):
        # Returns list of (sum, mask) where sum < L
        sums = [(0, 0)]
        for i, w in enumerate(weights_list):
            bit = 1 << i
            new_sums = [(s + w, m | bit) for s, m in sums if s + w < L]
            sums.extend(new_sums)
            if time.time() > deadline:
                break
        return sums

    sums_1 = gen_sums(weights_1)
    sums_2 = gen_sums(weights_2)
    
    # Sort sums_2 by sum value for binary search
    sums_2.sort(key=lambda x: x[0])
    sums_2_values = [s[0] for s in sums_2]
    
    best_sum = -1
    best_mask = 0
    
    for s1, m1 in sums_1:
        if time.time() > deadline:
            break
        # We need s2 such that s1 + s2 < L => s2 < L - s1
        target = L - s1 - 1
        idx = bisect.bisect_right(sums_2_values, target)
        if idx > 0:
            s2, m2 = sums_2[idx-1]
            total_sum = s1 + s2
            if total_sum > best_sum:
                best_sum = total_sum
                best_mask = m1 | (m2 << mid)
                
    # Convert best_mask to indices
    if best_mask == 0:
        return [0]
        
    result_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            result_indices.append(i)
    
    # Local search refinement: try 1-swap and 2-swap moves
    # to find a better packing
    current_indices = set(result_indices)
    current_sum = Fraction(sum(Fraction(1, d) for d in [denominators[i] for i in result_indices]), 1)
    
    # Precompute all fractions
    frac_values = [Fraction(1, d) for d in denominators]
    
    # Try local search with 1-swap moves
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        # Multiple restarts with different initial solutions
        random.seed(42)
        
        # Generate diverse initial solutions using greedy with random ordering
        for restart in range(50):
            if time.time() > deadline - 2.0:
                break
                
            # Greedy: add fractions in random order if they fit
            order = list(range(n))
            random.shuffle(order)
            
            greedy_set = set()
            greedy_sum = Fraction(0, 1)
            for idx in order:
                new_sum = greedy_sum + frac_values[idx]
                if new_sum < 1:
                    greedy_set.add(idx)
                    greedy_sum = new_sum
            
            # Local search on this greedy solution
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                # Try removing one element and adding another
                for remove_idx in list(greedy_set):
                    if time.time() > deadline - 2.0:
                        break
                    current_without = greedy_sum - frac_values[remove_idx]
                    for add_idx in range(n):
                        if add_idx in greedy_set:
                            continue
                        candidate_sum = current_without + frac_values[add_idx]
                        if candidate_sum < 1 and candidate_sum > greedy_sum:
                            greedy_set.remove(remove_idx)
                            greedy_set.add(add_idx)
                            greedy_sum = candidate_sum
                            improved = True
                            break
                    if time.time() > deadline - 2.0:
                        break
                
                # Also try pure additions
                if time.time() < deadline - 2.0:
                    for add_idx in range(n):
                        if add_idx not in greedy_set:
                            candidate_sum = greedy_sum + frac_values[add_idx]
                            if candidate_sum < 1 and candidate_sum > greedy_sum:
                                greedy_set.add(add_idx)
                                greedy_sum = candidate_sum
                                improved = True
                                break
                
                # Try pure removals if we're too close to 1 (shouldn't happen since we check < 1)
            
            # Compare with best found
            if greedy_sum > current_sum:
                current_sum = greedy_sum
                current_indices = greedy_set
    
    # Final result
    result_indices = sorted(current_indices)
    if not result_indices:
        return [0]
    return result_indices
